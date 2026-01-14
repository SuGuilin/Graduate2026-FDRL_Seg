import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from models.vim.models_mamba import VisionMamba
from FMTS_Fusion.mamba_simple_bak import Mamba
from einops import rearrange
import numbers
from typing import Optional, List
from einops.layers.torch import Rearrange


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = (
        x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, -1, H // window_size, W // window_size, window_size, window_size
    )
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return x


class LayerNorm_GASU(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm_moe(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = (
        int(in_batch / (r**2)),
        in_channel,
        r * in_height,
        r * in_width,
    )
    # 提取四种x分量
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch : out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2 : out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3 : out_batch * 4, :, :, :] / 2
    h = (
        torch.zeros([out_batch, out_channel, out_height, out_width])
        .float()
        .to(x.device)
    )

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class WMB(nn.Module):
    def __init__(self, channel, LayerNorm_type="WithBias"):
        super(WMB, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()
        self.norm1 = LayerNorm(channel, LayerNorm_type)
        self.process1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.norm2 = LayerNorm(channel, LayerNorm_type)
        self.GSAU = GSAU(channel)
        self.VisionMamba1 = SingleMambaBlock(channel)

    def forward(self, x):
        n, c, h, w = x.shape
        #print("WMB x:", x.shape)  # torch.Size([1, 192, 240, 320])
        shortcut = x
        x_norm1 = self.norm1(x)
        input_dwt = self.DWT(data_transform(x_norm1))
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
        #print("WMB input_LL:", input_LL.shape)  # torch.Size([1, 192, 120, 160])
        #print("WMB input_high:", input_high.shape)   # torch.Size([3, 192, 120, 160])

        B, C, H, W = input_high.shape
        high_rearrange = rearrange(input_high, "b c h w -> b (h w) c", h=H, w=W)
        #print("WMB high_rearrange:", high_rearrange.shape)  # torch.Size([3, 19200, 192])
        high_mamba = self.VisionMamba1(high_rearrange, H, W)
        output_high = high_mamba.view(B, C, H, W)

        output_LL = self.process1(input_LL)
        x_w = inverse_data_transform(
            self.IWT(torch.cat((output_LL, output_high), dim=0))
        )
        x_res = x_w + shortcut
        shortcut_2 = x_res
        output = shortcut_2 + self.GSAU(self.norm2(x_res))

        return output


class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm_GASU(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(
            dim,
            expand=1,
            d_state=8,
            bimamba_type="v6",
            if_devide_out=True,
            use_norm=True,
        )

    def forward(self, input, input_h=None, input_w=None):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)  # torch.Size([3, 19200, 192])
        #print("SingleMambaBlock:", input.shape)
        output = self.block(input, input_h, input_w)
        return output + skip


class DiffEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(DiffEnhance, self).__init__()
        self.local_attention_tde = nn.Sequential(
            nn.Conv2d(1, 4 * reduction, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(4, 4 * reduction), 
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.local_attention_rse = nn.Sequential(
            nn.Conv2d(1, 4 * reduction, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(4, 4 * reduction), 
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_attention_tde = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.global_attention_rse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, tir, mode="tde"):
        if mode == "tde":
            return self.forward_tde(rgb, tir)
        elif mode == "rse":
            return self.forward_rse(rgb, tir)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_tde(self, rgb, tir):
        rgb_max_out, _ = torch.max(rgb, dim=1, keepdim=True)
        local_mask = self.local_attention_tde(rgb_max_out)
        global_mask = self.global_attention_tde(tir)

        tir_enhanced = tir * local_mask * global_mask

        fused_input = torch.cat((rgb, tir), dim=1)
        fused_features = self.feature_transform(fused_input)
        fused_features = fused_features * local_mask + tir_enhanced
        return fused_features

    def forward_rse(self, rgb, tir):
        tir_max_out, _ = torch.max(tir, dim=1, keepdim=True)
        local_mask = self.local_attention_rse(tir_max_out)

        rgb_refined = rgb * local_mask
        tir_refined = tir * local_mask + rgb_refined

        rgb_enhanced = self.global_attention_rse(rgb_refined) * rgb_refined
        tir_enhanced = self.global_attention_rse(tir_refined) * tir_refined

        return rgb_enhanced + tir_enhanced



class Mamba_Block(nn.Module):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.WMB = WMB(channel)

    def forward(self, x):
        output = self.WMB(x)
        return output



#################################################################################
## Multi-Modal Feature Enhancement Module

class MoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        lr_space: str = "linear",
        recursive: int = 2,
    ):
        super().__init__()
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0),
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, padding=2, groups=in_ch), nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True), 
            #nn.LayerNorm(in_ch),
            #Mamba(in_ch, d_state=64, bimamba_type=None),
            nn.GELU()
        )

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[
                Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)
            ],  # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.conv_1(x)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        # normalize the weights of the selected experts
        # topk_weights = F.softmax(topk_weights, dim=1, dtype=torch.float).to(inputs.dtype)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i : i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i : i + 1, None, None]

        return out

# Reference: https://github.com/eduardzamfir/seemoredetails/blob/main/basicsr/archs/seemore_arch.py
class Expert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        low_dim: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StripedConv2d(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                groups=in_ch if depthwise else 1,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                groups=in_ch if depthwise else 1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedFFN(nn.Module):
    def __init__(self, in_ch, mlp_ratio, kernel_size, act_layer,):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio
        
        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )
        
        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2, 
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)
    
    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        
        gate = self.gate(gate)
        x = x * gate
        
        x = self.fn_2(x)
        return x



class ModalFusion(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__() 
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced_dim = max(channels * 2 // reduction_ratio, 1)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels * 2, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels * 2),
            nn.Sigmoid()
        )
        
        self.conv = nn.Conv2d(channels * 2, channels , 1)

    def forward(self, rgb, ir):
        rgb_w = self.gap(rgb)
        ir_w = self.gap(ir)
        
        w = torch.cat([rgb_w, ir_w], dim=1)
        w = self.channel_attention(w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        w1, w2 = torch.chunk(w, chunks=2, dim=1)

        fuse = torch.cat([w1* rgb, w2 * ir], dim=1)
        return self.conv(fuse)


class ResMoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        lr_space: int = 2,
        recursive: int = 2,
    ):
        super().__init__()
        lr_space_mapping = {1: "linear", 2: "exp", 3: "double"}
        self.fusion = ModalFusion(channels=in_ch)
        #self.before = nn.Conv2d(2 * in_ch, 2 * in_ch, 3, 1, 1)
        #self.fusion_conv = nn.Conv2d(2 * in_ch, in_ch, 1, 1)
        #self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = LayerNorm_moe(in_ch, data_format="channels_first")
        self.block = MoEBlock(
           in_ch=in_ch,
           num_experts=num_experts,
           topk=topk,
           recursive=recursive,
           lr_space=lr_space_mapping.get(lr_space, "linear"),
        )
        self.norm_2 = LayerNorm_moe(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, rgb, ir):
        x = self.fusion(rgb, ir)
        #x = self.fusion_conv(self.lrelu(self.before(torch.cat([rgb, ir], dim=1))))
        x = self.block(self.norm(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x



class SGLMamba(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        gap_pad = self.avg_pool(x_Mamba_pad)
        gmp_pad = self.max_pool(x_Mamba_pad)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops



class SGLMamba1(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba1, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        #if in_chans == 3 or in_chans == 6:
        #    rgb_mean = (0.4488, 0.4371, 0.4040)
        #    rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
        #    self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        #    self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        #else:
        #    pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        #self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        #self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = x_AB_conv #self.mamba_AB1(x_AB_conv)
        x_BA_1 = x_BA_conv #self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops



class SGLMamba_baseline(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_baseline, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        #self.mamba_A1 = Mamba_Block(channel=N)
        #self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        #self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        #self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        #self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        #self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        #self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        #self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        #self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        #self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        #self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.before = nn.Conv2d(2 * N, 2 * N, 3, 1, 1)
        self.fusion_conv = nn.Conv2d(2 * N, N, 1, 1)

        #self.fc = nn.Sequential(
        #        nn.Linear(embed_dim*4, embed_dim*4 // 4),
        #        nn.ReLU(inplace=True),
        #        nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        #x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        #x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        #x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        #x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        #x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        #x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        #x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        #x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        #x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        #x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        #x_A_Mamba_3_temp = x_A_Mamba_3
        #x_B_Mamba_3_temp = x_B_Mamba_3

        #x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ###print("pad:", x_Mamba_pad.shape)
        #gap_pad = self.avg_pool(x_Mamba_pad)
        ###print("gap_pad:", gap_pad.shape)
        #gmp_pad = self.max_pool(x_Mamba_pad)
        ###print("gmp_pad:", gmp_pad.shape)
        #gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        #w = self.fc(gp_pad).permute(0, 3, 1, 2)
        #w = self.sigmoid_final(w)
        ##print(w.shape)


        #x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        #x_AB_conv = x_AB_conv * w
        #x_AB_conv = self.conv_cat_A(x_AB_conv)
        #x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        #x_BA_conv = x_BA_conv * (1 - w)
        #x_BA_conv = self.conv_cat_B(x_BA_conv)

        #x_AB_1 = self.mamba_AB1(x_AB_conv)
        #x_BA_1 = self.mamba_BA1(x_BA_conv)

        #x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        #x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        #fuse = self.moe(x_AB_1, x_BA_1)

        fuse = self.fusion_conv(self.lrelu(self.before(torch.cat([x_A_temp, x_B_temp], dim=1))))
        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops


class SGLMamba_model1(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model1, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        #x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops


class SGLMamba_model2(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model2, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        #x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops

class SGLMamba_model6(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model6, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops

class SGLMamba_model5(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model5, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)
        #self.before = nn.Conv2d(2 * N, 2 * N, 3, 1, 1)
        #self.fusion_conv = nn.Conv2d(2 * N, N, 1, 1)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp
        #print(x_AB_1.shape)

        fuse = self.moe(x_AB_1, x_BA_1)
        #fuse = self.fusion_conv(self.lrelu(self.before(torch.cat([x_AB_1, x_BA_1], dim=1))))

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops

class SGLMamba_model4(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model4, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        #self.mamba_A1 = Mamba_Block(channel=N)
        #self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = x_A_temp #self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = x_B_temp #self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops


class SGLMamba_model3(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model3, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops


class SGLMamba_model3(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba_model3, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        #self.mamba_A2 = Mamba_Block(channel=N)
        #self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        #self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        #self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)

        x_A_Mamba_1 = x_A_Mamba_1 + self.DiffEnhance(x_A_Mamba_1, x_B_Mamba_1, "rse")

        x_A_Mamba_3 = x_A_Mamba_1 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_1 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_1 = x_AB_1 + x_A_Mamba_3_temp
        x_BA_1 = x_BA_1 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_1, x_BA_1)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops



class SGLMamba2(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
        super(SGLMamba2, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.DiffEnhance = DiffEnhance(embed_dim, embed_dim)

        N = embed_dim
    
        self.mamba_A1 = Mamba_Block(channel=N)
        self.mamba_B1 = Mamba_Block(channel=N)
        self.mamba_A2 = Mamba_Block(channel=N)
        self.mamba_B2 = Mamba_Block(channel=N)
        self.mamba_AB1 = Mamba_Block(channel=N)
        self.mamba_AB2 = Mamba_Block(channel=N)
        self.mamba_BA1 = Mamba_Block(channel=N)
        self.mamba_BA2 = Mamba_Block(channel=N)

        self.moe = ResMoEBlock(in_ch=embed_dim, num_experts=4, topk=2)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale=2, num_feat=N, num_out_ch=N)
        self.conv_re_1 = nn.Conv2d(N, N, 3, 1, 1)
        self.conv_re_2 = nn.Conv2d(N, 64, 3, 1, 1)
        self.conv_re_3 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv_cat_A = nn.Conv2d(2 * N, N, 1, 1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 4 // 4, embed_dim * 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B))  # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))
        
        x_B_temp = x_B_temp + self.DiffEnhance(x_A_temp, x_B_temp, "tde")

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_A_Mamba_2 = self.mamba_A2(x_A_Mamba_1)
        x_B_Mamba_1 = self.mamba_B1(x_B_temp)
        x_B_Mamba_2 = self.mamba_B2(x_B_Mamba_1)

        x_A_Mamba_2 = x_A_Mamba_2 + self.DiffEnhance(x_A_Mamba_2, x_B_Mamba_2, "rse")

        x_A_Mamba_3 = x_A_Mamba_2 + x_A_temp  # (B,192,H/2,W/2)
        x_B_Mamba_3 = x_B_Mamba_2 + x_B_temp  # (B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_Mamba_pad = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)

        ##print("pad:", x_Mamba_pad.shape)
        gap_pad = self.avg_pool(x_Mamba_pad)
        ##print("gap_pad:", gap_pad.shape)
        gmp_pad = self.max_pool(x_Mamba_pad)
        ##print("gmp_pad:", gmp_pad.shape)
        gp_pad = torch.cat([gap_pad, gmp_pad], dim=1).permute(0, 2, 3, 1)
        w = self.fc(gp_pad).permute(0, 3, 1, 2)
        w = self.sigmoid_final(w)
        #print(w.shape)


        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = x_AB_conv * w
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = x_BA_conv * (1 - w)
        x_BA_conv = self.conv_cat_B(x_BA_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_2 = self.mamba_AB2(x_AB_1)
        x_BA_2 = self.mamba_BA2(x_BA_1)

        x_AB_2 = x_AB_2 + x_A_Mamba_3_temp
        x_BA_2 = x_BA_2 + x_B_Mamba_3_temp

        fuse = self.moe(x_AB_2, x_BA_2)

        # [B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = self.upsample(fuse)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))
        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))
        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops



if __name__ == "__main__":
    upscale = 4
    window_size = 8
    height = 480  # (480 // upscale // window_size + 1) * window_size
    width = 640  # (640 // upscale // window_size + 1) * window_size
    model = SGLMamba().to('cuda:2')
    print("h:", height)
    print("w:", width)
    # print(model)
    x = torch.randn((2, 3, height, width)).to('cuda:2')
    y = torch.randn((2, 3, height, width)).to('cuda:2')
    z = model(x, y)
    print(z.shape)




