import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return sobelx, sobely


#class FusionLoss(nn.Module):
#    def __init__(self):
#        super(FusionLoss, self).__init__()
#        self.l1_loss =  nn.L1Loss()
#        #self.cfg = cfg

#    def forward(self, input_vis, input_ir, Fuse):#input_xy, output, Mask):
#        #input_vis, input_ir = input_xy 

#        #Fuse = output * Mask
#        YCbCr_Fuse = RGB2YCrCb(Fuse) 
#        Y_Fuse  = YCbCr_Fuse[:,0:1,:,:]
#        Cr_Fuse = YCbCr_Fuse[:,1:2,:,:]
#        Cb_Fuse = YCbCr_Fuse[:,2:,:,:]  
        

#        R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 0.7, 1)
#        YCbCr_R_vis = RGB2YCrCb(R_vis) 
#        Y_R_vis = YCbCr_R_vis[:,0:1,:,:]
#        Cr_R_vis = YCbCr_R_vis[:,1:2,:,:]
#        Cb_R_vis = YCbCr_R_vis[:,2:,:,:]          
        
                        
#        R_ir = torchvision.transforms.functional.adjust_contrast(input_ir, 1.3)


#        Fuse_R = torch.unsqueeze(Fuse[:,0,:,:],1)
#        Fuse_G = torch.unsqueeze(Fuse[:,1,:,:],1)
#        Fuse_B = torch.unsqueeze(Fuse[:,2,:,:],1)
#        Fuse_R_grad_x,Fuse_R_grad_y =   Sobelxy(Fuse_R)
#        Fuse_G_grad_x,Fuse_G_grad_y =   Sobelxy(Fuse_G)
#        Fuse_B_grad_x,Fuse_B_grad_y =   Sobelxy(Fuse_B)
#        Fuse_grad_x = torch.cat([Fuse_R_grad_x, Fuse_G_grad_x, Fuse_B_grad_x], 1)
#        Fuse_grad_y = torch.cat([Fuse_R_grad_y, Fuse_G_grad_y, Fuse_B_grad_y], 1)


#        R_VIS_R = torch.unsqueeze(R_vis[:,0,:,:],1)
#        R_VIS_G = torch.unsqueeze(R_vis[:,1,:,:],1)
#        R_VIS_B = torch.unsqueeze(R_vis[:,2,:,:],1)
#        R_VIS_R_grad_x, R_VIS_R_grad_y =   Sobelxy(R_VIS_R)
#        R_VIS_G_grad_x, R_VIS_G_grad_y =   Sobelxy(R_VIS_G)
#        R_VIS_B_grad_x, R_VIS_B_grad_y =   Sobelxy(R_VIS_B)
#        R_VIS_grad_x = torch.cat([R_VIS_R_grad_x, R_VIS_G_grad_x, R_VIS_B_grad_x], 1)
#        R_VIS_grad_y = torch.cat([R_VIS_R_grad_y, R_VIS_G_grad_y, R_VIS_B_grad_y], 1)


#        R_IR_R = torch.unsqueeze(R_ir[:,0,:,:],1)
#        R_IR_G = torch.unsqueeze(R_ir[:,1,:,:],1)
#        R_IR_B = torch.unsqueeze(R_ir[:,2,:,:],1)
#        R_IR_R_grad_x,R_IR_R_grad_y =   Sobelxy(R_IR_R)
#        R_IR_G_grad_x,R_IR_G_grad_y =   Sobelxy(R_IR_G)
#        R_IR_B_grad_x,R_IR_B_grad_y =   Sobelxy(R_IR_B)
#        R_IR_grad_x = torch.cat([R_IR_R_grad_x, R_IR_G_grad_x,R_IR_B_grad_x], 1)
#        R_IR_grad_y = torch.cat([R_IR_R_grad_y, R_IR_G_grad_y,R_IR_B_grad_y], 1)


#        joint_grad_x = torch.maximum(R_VIS_grad_x, R_IR_grad_x)
#        joint_grad_y = torch.maximum(R_VIS_grad_y, R_IR_grad_y)
#        joint_int  = torch.maximum(R_vis, R_ir)
        
        
#        con_loss = self.l1_loss(Fuse, joint_int)
#        gradient_loss = 0.5 * self.l1_loss(Fuse_grad_x, joint_grad_x) + 0.5 * self.l1_loss(Fuse_grad_y, joint_grad_y)
#        color_loss = self.l1_loss(Cb_Fuse, Cb_R_vis) + self.l1_loss(Cr_Fuse, Cr_R_vis)


#        fusion_loss_total = 3.0 * con_loss  + 10.0 * gradient_loss  + 5.0 * color_loss

#        return fusion_loss_total, gradient_loss, con_loss



class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # ### NEW 权重
        self.y_contrast_weight = 2.0

    # ### NEW 对比度损失：Y 通道标准差
    def y_contrast_loss(self, y_fuse, y_vis, y_ir):
        std_f = torch.std(y_fuse)
        std_v = torch.std(y_vis)
        std_i = torch.std(y_ir)
        target = torch.maximum(std_v, std_i)
        return F.relu(target - std_f + 0.01)  # 只惩罚低于目标

    def forward(self, input_vis, input_ir, Fuse):
        # ---------- 预处理 ----------
        R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 0.7, 1)
        R_ir  = torchvision.transforms.functional.adjust_contrast(input_ir, 1.3)

        #R_vis = input_vis
        #R_ir  = input_ir

        # ---------- YCbCr ----------
        YCbCr_Fuse    = RGB2YCrCb(Fuse)
        Y_Fuse, Cr_Fuse, Cb_Fuse = YCbCr_Fuse[:, 0:1], YCbCr_Fuse[:, 1:2], YCbCr_Fuse[:, 2:3]

        YCbCr_R_vis   = RGB2YCrCb(R_vis)
        Y_R_vis, Cr_R_vis, Cb_R_vis = YCbCr_R_vis[:, 0:1], YCbCr_R_vis[:, 1:2], YCbCr_R_vis[:, 2:3]

        # ### NEW 计算红外 Y 通道
        y_ir = 0.299 * R_ir[:, 0:1] + 0.587 * R_ir[:, 1:2] + 0.114 * R_ir[:, 2:3]

        # ---------- 梯度 ----------
        def grad_rgb(x):
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            rx, ry = Sobelxy(r)
            gx, gy = Sobelxy(g)
            bx, by = Sobelxy(b)
            return torch.cat([rx, gx, bx], 1), torch.cat([ry, gy, by], 1)

        Fuse_grad_x, Fuse_grad_y = grad_rgb(Fuse)
        R_VIS_grad_x, R_VIS_grad_y = grad_rgb(R_vis)
        R_IR_grad_x,  R_IR_grad_y  = grad_rgb(R_ir)

        joint_grad_x = torch.maximum(R_VIS_grad_x, R_IR_grad_x)
        joint_grad_y = torch.maximum(R_VIS_grad_y, R_IR_grad_y)
        joint_int    = torch.maximum(R_vis, R_ir)

        # ---------- 原损失 ----------
        con_loss = self.l1_loss(Fuse, joint_int)
        gradient_loss = 0.5 * (self.l1_loss(Fuse_grad_x, joint_grad_x) +
                               self.l1_loss(Fuse_grad_y, joint_grad_y))

        # ### NEW 归一化颜色损失
        color_loss = self.l1_loss(Cb_Fuse, Cb_R_vis) + self.l1_loss(Cr_Fuse, Cr_R_vis)
        # ### NEW Y 通道对比度损失
        y_cont_loss = self.y_contrast_loss(Y_Fuse, Y_R_vis, y_ir)

        # ---------- 总损失 ----------
        fusion_loss_total = (10.0 * con_loss +
                             5.0 * gradient_loss +
                             15.0 * color_loss +
                             10.0 * y_cont_loss)
        return fusion_loss_total, gradient_loss, con_loss







#class L_Grad(nn.Module):
#    def __init__(self):
#        super(L_Grad, self).__init__()
#        self.sobelconv=Sobelxy()

#    def forward(self, image_A, image_B, image_fused):
#        print(image_A.shape)
#        print(image_B.shape)
#        print(image_fused.shape)
#        gradient_A = self.sobelconv(image_A)
#        gradient_B = self.sobelconv(image_B)
#        gradient_fused = self.sobelconv(image_fused)
#        gradient_joint = torch.maximum(gradient_A, gradient_B)
#        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
#        return Loss_gradient

#class Sobelxy(nn.Module):
#    def __init__(self):
#        super(Sobelxy, self).__init__()
#        kernelx = [[-1, 0, 1],
#                  [-2,0 , 2],
#                  [-1, 0, 1]]
#        kernely = [[1, 2, 1],
#                  [0,0 , 0],
#                  [-1, -2, -1]]
#        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#    def forward(self,x):
#        sobelx=F.conv2d(x, self.weightx, padding=1)
#        sobely=F.conv2d(x, self.weighty, padding=1)
#        return torch.abs(sobelx)+torch.abs(sobely)

#class L_Intensity(nn.Module):
#    def __init__(self):
#        super(L_Intensity, self).__init__()

#    def forward(self, image_A, image_B, image_fused):
#        intensity_joint = torch.max(image_A, image_B)
#        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#        return Loss_intensity

#def int_loss (fused_result,input_ir,input_vi):
#    ir_loss=torch.mean(torch.square(fused_result-input_ir))
#    vi_loss=torch.mean(torch.square(fused_result-input_vi))
#    return ir_loss+vi_loss

#class fusion_loss_vif(nn.Module):
#    def __init__(self):
#        super(fusion_loss_vif, self).__init__()
#        self.L_Grad = L_Grad()
#        self.L_Inten = L_Intensity()

#    def forward(self, image_A, image_B, image_fused):
#        loss_int = 10 * self.L_Inten(image_A, image_B, image_fused)
#        loss_gradient = 1 * self.L_Grad(image_A, image_B, image_fused)
#        fusion_loss = loss_int+ loss_gradient
#        return fusion_loss, loss_gradient, loss_int

