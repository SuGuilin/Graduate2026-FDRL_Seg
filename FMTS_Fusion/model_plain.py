from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import argparse

from select_network import define_G
from model_base import ModelBase
import os

from torch.utils.tensorboard import SummaryWriter
from utils1.utils_model import test_mode
from utils1.utils_regularizers import regularizer_orth, regularizer_clip
import clip
import torch.nn.functional as F

def smart_truncate_text(text, max_tokens=77):
    """智能截断文本，在句子边界处断开"""
    # 先尝试简单的单词截断
    words = text.split()
    if len(words) <= max_tokens:
        return text
    
    # 截断到最大token数
    truncated = ' '.join(words[:max_tokens])
    
    # 尝试在句子边界处断开
    last_period = truncated.rfind('.')
    last_comma = truncated.rfind(',')
    
    # 优先在句号处断开
    if last_period > len(truncated) * 0.6:  # 如果句号在文本的后40%
        return truncated[:last_period + 1]
    # 其次在逗号处断开
    elif last_comma > len(truncated) * 0.7:
        return truncated[:last_comma]
    else:
        # 在单词边界处断开并添加省略号
        return truncated + '...'

class CLIPPreprocessor:
    def __init__(self, target_size=224):
        self.target_size = target_size
        # CLIP 的 ImageNet 标准化参数
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    
    def __call__(self, images):
        """  
        Args:
            images: [B, C, H, W] 在 [0, 1] 范围
            
        Returns:
            预处理后的图像 [B, C, 224, 224]
        """
        # 调整尺寸
        if images.shape[2:] != (self.target_size, self.target_size):
            images = F.interpolate(
                images, 
                size=(self.target_size, self.target_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 标准化
        mean = self.mean.view(1, 3, 1, 1).to(images.device)
        std = self.std.view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        return images


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        self.clip_model, self.preprocess = clip.load("ViT-B/32")
        self.preprocessor = CLIPPreprocessor()

        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        # ------------------------------------
        # Define Tensorboard 
        # ------------------------------------
        #创建tensorboard路径 利用summarywriter记录训练过程中的数据
        tensorboard_path = os.path.join(self.opt['path']['tensorboard'], 'Tensorboard')
        print(tensorboard_path)
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.step_counter = 0
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'vif':
            from models.loss_vif import FusionLoss
            self.G_lossfn = FusionLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']


    # ----------------------------------------
    # clip loss
    # ----------------------------------------
    def clip_loss(self):
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        processed_texts = []
        for text in self.text_fusion[0]:
            tokens = clip.tokenize(text, context_length=77, truncate=True)
            if tokens.shape[-1] > 77:
                text = clip.decode(clip.tokenize(text)[:, :77])
            processed = smart_truncate_text(text)
            processed_texts.append(processed)
    
        with torch.no_grad():
            #print("vi:", is_in_01_range(self.A))
            #print("ir:", is_in_01_range(self.B))
            #print("fusion:", is_in_01_range(self.E))
            vi_processed = self.preprocessor(self.A)
            ir_processed = self.preprocessor(self.B)
            fusion_processed = self.preprocessor(self.E)
            vi_features = self.clip_model.encode_image(vi_processed)
            ir_features = self.clip_model.encode_image(ir_processed)
            fusion_features = self.clip_model.encode_image(fusion_processed)

            text_input = clip.tokenize(processed_texts, context_length=77, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_input)
        
        # 计算相似度损失 (文本引导损失)
        similarities = (text_features @ fusion_features.T).squeeze(0)
        similarities = torch.sigmoid(similarities)
        loss_text_guidance = 1 - similarities.mean()
        
        # 计算特征保持损失
        loss_feature_preservation = (
            torch.mean(torch.square(fusion_features - ir_features)) + 
            torch.mean(torch.square(fusion_features - vi_features))
        )
        
        # 总 CLIP 损失
        total_clip_loss = loss_text_guidance + loss_feature_preservation

        return total_clip_loss


    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.CosineAnnealingLR(self.G_optimizer,T_max=self.opt_train['epoch']))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed under/over data
    # ----------------------------------------
    def feed_data(self, data, need_GT=False, phase='test'):
        self.A = data['A'].to(self.device)
        self.B = data['B'].to(self.device)
        self.text_fusion = data['text_fusion']
        #print(self.text_fusion)
        #print(self.B.shape)
        if need_GT:
            self.GT = data['GT'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, phase='test'):
        self.E = self.netG(self.A, self.B)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        print(f"[Debug] Step {current_step} - Optimizing...")

        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_lossfn_type = self.opt_train['G_lossfn_type']
        ## loss function
        if G_lossfn_type in ['vif']:
            total_loss, loss_grad, loss_int= self.G_lossfn(self.A, self.B, self.E)
            clip_loss = self.clip_loss()
            G_loss = self.G_lossfn_weight * total_loss + clip_loss * 10.0
            print(f"[Debug] Fusion Loss: {G_loss.item()} | Text: {clip_loss.item()} | Int: {loss_int.item()}")
        else:
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.GT)
            print(f"[Debug] Fusion Loss (other): {G_loss.item()}")
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)
        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['Fusion_loss'] = G_loss.item()
        if G_lossfn_type in ['loe', 'mef', 'vif', 'mff', 'gt', 'nir', 'med']:
            self.log_dict['Text_loss'] = clip_loss.item()
            self.log_dict['Int_loss'] = loss_int.item()

        self.writer.add_scalar('Loss/Fusion_loss', self.log_dict['Fusion_loss'], current_step)
        self.writer.add_scalar('Loss/Text_loss', self.log_dict['Text_loss'], current_step)
        self.writer.add_scalar('Loss/Int_loss', self.log_dict['Int_loss'], current_step)

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

        #记录训练过程中的图像
        if G_lossfn_type == 'vif':
            if self.step_counter % 100 == 0:
                self.writer.add_image('ir_image', self.A[0],global_step=self.step_counter)
                self.writer.add_image('vi_image', self.B[0],global_step=self.step_counter)
                self.writer.add_image('fused_image', self.E[0],global_step=self.step_counter)
            self.step_counter += 1

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict
