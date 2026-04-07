"""
Created on 2025/6/4 15:18

@author: Minghan Li
we have 3 forward model: CSLIP CLIP Coded Aperture
"""
import torch
from torch import nn
from utils import *

#===========================CSLIP model============================
def shift_tensor_batch(tensor, shifts_x, shifts_y):
    """
    批量平移张量，每个样本独立平移
    tensor: (B, C, H, W)
    shifts_x: (B,) 水平平移量（右正左负）
    shifts_y: (B,) 垂直平移量（下正上负）
    """
    B, C, H, W = tensor.shape
    device = tensor.device

    # 生成基础网格
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, W, device=device),
        torch.linspace(-1, 1, H, device=device),
        indexing='xy'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, H, W, 2)

    # 计算归一化偏移（反转方向）
    dx = (-shifts_x.float() * (2.0 / W)  )
    dy = (-shifts_y.float() * (2.0 / H)  )

    # 应用偏移
    grid = grid.clone()
    grid[..., 0] += dx.view(B, 1, 1)
    grid[..., 1] += dy.view(B, 1, 1)

    # 执行采样
    return F.grid_sample(
        tensor,
        grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
def forward_compute_proj_batch(lf,code_batch):
    if len(lf.shape)==4:
        nu,nv,H,W = lf.shape
        is_rgb = False
    elif len(lf.shape)==5:
        nu,nv,c,H,W = lf.shape
        is_rgb = True
    batch_size = code_batch.shape[0]
    
    if is_rgb:
        # RGB: code_batch [batch, H, W] -> [batch, 1, 1, 1, H, W]
        # lf [nu, nv, 3, H, W] -> [1, nu, nv, 3, H, W]
        lf_coded = code_batch.unsqueeze(1).unsqueeze(1).unsqueeze(1) * lf.unsqueeze(0)  # [batch, nu, nv, 3, H, W]
    else:
        lf_coded = code_batch.unsqueeze(1).unsqueeze(1) * lf.unsqueeze(0)  # [batch, nu, nv, H, W]

    B_shift = nu * nv
    shifts_x = (torch.arange(nu, device=lf.device).view(nu, 1).expand(nu, nv).reshape(B_shift) - nu // 2).repeat(
        batch_size)
    shifts_y = (torch.arange(nv, device=lf.device).view(1, nv).expand(nu, nv).reshape(B_shift) - nv // 2).repeat(
        batch_size)
    
    if is_rgb:
        # RGB: reshape to [batch*B_shift, 3, H, W]
        shifted_flat = shift_tensor_batch(lf_coded.reshape(batch_size * B_shift, c, H, W), shifts_x, shifts_y)
        lf_proj_hat_batch = shifted_flat.view(batch_size, B_shift, c, H, W).sum(dim=1) / (nu * nv)  # [batch, 3, H, W]
    else:
        shifted_flat = shift_tensor_batch(lf_coded.reshape(batch_size * B_shift, 1, H, W), shifts_x, shifts_y)
        lf_proj_hat_batch = shifted_flat.view(batch_size, B_shift, 1, H, W).sum(dim=1).squeeze(1) / (nu * nv)  # [batch, H, W]
    
    return lf_proj_hat_batch.squeeze()
def H_ForwardOperator_opt(C, LF, batch_size=10,usecheckpoint=False):
    num = C.shape[0]
    is_rgb = LF.dim() == 5
    
    if is_rgb:
        _, _, c, H, W = LF.shape
        lf_proj_hat = torch.zeros(num, c, H, W, device=LF.device)
    else:
        H, W = LF.shape[2:]
        lf_proj_hat = torch.zeros(num, H, W, device=LF.device)
    
    for start in range(0, num, batch_size):
        end = min(start + batch_size, num)
        code_batch = C[start:end].to(LF.device, non_blocking=True)
        if usecheckpoint:
            lf_proj_hat[start:end] = checkpoint(forward_compute_proj_batch, LF, code_batch, use_reentrant=False)
        else:
            lf_proj_hat[start:end] = forward_compute_proj_batch(LF, code_batch)
    return lf_proj_hat
def adjoint_compute_batch(batch_proj, batch_C, shifts_x, shifts_y, nu, nv):
    """
    处理一个批次的伴随算子计算
    batch_proj: (batch_size, H, W) 或 (batch_size, 3, H, W) for RGB
    batch_C: (batch_size, H, W)
    shifts_x: (nu * nv,)
    shifts_y: (nu * nv,)
    nu, nv: 光场角分辨率
    """
    batch_size_actual = batch_proj.shape[0]
    device = batch_proj.device
    is_rgb = batch_proj.dim() == 4
    
    if is_rgb:
        _, c, H, W = batch_proj.shape
    else:
        H, W = batch_proj.shape[1:]
    
    # 扩展 batch_proj 以匹配 (batch_size, nu * nv, c, H, W) 或 (batch_size, nu * nv, H, W)
    if is_rgb:
        batch_proj_exp = batch_proj.unsqueeze(1).repeat(1, nu * nv, 1, 1, 1)  # (batch_size, nu * nv, 3, H, W)
    else:
        batch_proj_exp = batch_proj.unsqueeze(1).repeat(1, nu * nv, 1, 1)  # (batch_size, nu * nv, H, W)

    # 准备平移量
    shifts_x_batch = shifts_x.repeat(batch_size_actual)  # (batch_size * nu * nv,)
    shifts_y_batch = shifts_y.repeat(batch_size_actual)  # (batch_size * nu * nv,)

    # 平移投影
    if is_rgb:
        batch_proj_flat = batch_proj_exp.view(batch_size_actual * nu * nv, c, H, W)
        shifted_proj_flat = shift_tensor_batch(batch_proj_flat, shifts_x_batch, shifts_y_batch)
        shifted_proj = shifted_proj_flat.view(batch_size_actual, nu * nv, c, H, W)  # (batch_size, nu * nv, 3, H, W)
        
        # 重塑 batch_C 以进行广播
        batch_C_reshaped = batch_C.unsqueeze(1).unsqueeze(1).expand(batch_size_actual, nu * nv, c, H, W)  # (batch_size, nu * nv, 3, H, W)
    else:
        batch_proj_flat = batch_proj_exp.view(batch_size_actual * nu * nv, 1, H, W)
        shifted_proj_flat = shift_tensor_batch(batch_proj_flat, shifts_x_batch, shifts_y_batch)
        shifted_proj = shifted_proj_flat.view(batch_size_actual, nu * nv, 1, H, W)  # (batch_size, nu * nv, 1, H, W)
        
        # 重塑 batch_C 以进行广播
        batch_C_reshaped = batch_C.unsqueeze(1).expand(batch_size_actual, nu * nv, H, W).unsqueeze(2)  # (batch_size, nu * nv, 1, H, W)

    # 加权并求和
    weighted_shifted = shifted_proj * batch_C_reshaped
    if is_rgb:
        sum_weighted = weighted_shifted.sum(dim=0).view(nu, nv, c, H, W)
    else:
        sum_weighted = weighted_shifted.sum(dim=0).view(nu, nv, H, W)

    return sum_weighted
def H_AdjointOperator_opt(LF_proj, C, nu, nv, batch_size=10):
    """
    优化后的伴随算子，使用 checkpoint 减少内存占用
    LF_proj: (num, H, W) 或 (num, 3, H, W) for RGB 投影图像张量
    C: (num, H, W) 编码张量
    nu, nv: 光场角分辨率
    batch_size: 批处理大小
    返回: LF_adj (nu, nv, H, W) 或 (nu, nv, 3, H, W) for RGB 伴随光场张量
    """
    num = LF_proj.shape[0]
    device = LF_proj.device
    is_rgb = LF_proj.dim() == 4
    
    if is_rgb:
        _, c, H, W = LF_proj.shape
        LF_adj = torch.zeros(nu, nv, c, H, W, dtype=LF_proj.dtype, device=device)
    else:
        H, W = LF_proj.shape[1:]
        LF_adj = torch.zeros(nu, nv, H, W, dtype=LF_proj.dtype, device=device)
    
    u0 = nu // 2
    v0 = nv // 2

    # 准备所有 (u, v) 组合的平移量
    u_range = torch.arange(nu, device=device) - u0
    v_range = torch.arange(nv, device=device) - v0
    shifts_x = -u_range.view(nu, 1).expand(nu, nv).reshape(nu * nv)  # (nu * nv,)
    shifts_y = -v_range.view(1, nv).expand(nu, nv).reshape(nu * nv)  # (nu * nv,)

    for start in range(0, num, batch_size):
        end = min(start + batch_size, num)
        batch_proj = LF_proj[start:end]
        batch_C = C[start:end].to(device, non_blocking=True)

        # 使用 checkpoint 调用 adjoint_compute_batch
        sum_weighted = checkpoint(adjoint_compute_batch, batch_proj, batch_C, shifts_x, shifts_y, nu, nv, use_reentrant=False)
        LF_adj += sum_weighted

    LF_adj /= (nu * nv)
    return LF_adj


#===========================CLIP model============================
class CLIP_SPC(nn.Module):
    def __init__(self, codes):
        super().__init__()
        self.codes = codes
        u, v, h, w, np = codes.shape
        self.norm_fac = 1 / (h * w)

    def forward(self, LF_in):
        '''  The forward model for Single Pixel Light Field camera based on CLIP
        #  1 codes: codes used in SPC: [Na, Na, H, W, N_pattern], should be cuda device
        #  2 LF_in: cuda - float, shaped in [Na, Na, H, W], Na: angular resolution
        '''
        # [Na, Na, H, W, N_pattern] * [Na, Na, H, W ]   -->  # [Na, Na, N_pattern]
        SPC_data = torch.einsum('uvmnp,uvmn->uvp', self.codes, LF_in)
        return SPC_data  # * self.norm_fac # [Na, Na, N_pattern]

    def ajoint(self, spc_data_in):
        '''  The adjoint model for Single Pixel Light Field camera based on CLIP
        #  1 codes: codes used in SPC: [Na, Na, H, W, N_pattern], should be cuda device
        #  2 spc_data_in: the spc measurement, cuda-float, shaped in [u, v, Npattern],
        '''
        # [u, v, h, w, np] * [u, v, 1, 1, np]  --> [u, v, h, w, np] and then sum along np
        u, v, Npattern = spc_data_in.shape
        lf_recon = torch.einsum('uvhwp,uvhwp->uvhw', self.codes, spc_data_in.unsqueeze(2).unsqueeze(3))
        return lf_recon  # *self.norm_fac #  [u, v, h, w]

    def CLIP_fwd(self, im_in):
        '''  The forward model for Single Pixel Light Field camera based on CLIP
        #  1 codes: codes used in SPC: [Na, Na, H, W, N_pattern], should be cuda device
        #  2 LF_in: cuda - float, shaped in [Na, Na, H, W], Na: angular resolution
        '''
        # [Na, Na, H, W, N_pattern] * [1, 1, H, W ]   -->  # [Na, Na, N_pattern]
        SPC_data = torch.einsum('uvmnp,uvmn->uvp', self.codes, im_in.unsqueeze(0).unsqueeze(0))
        return SPC_data  # * self.norm_fac # [Na, Na, N_pattern]

def CLIP_SP_Adjoint(codes, spc_data_in):
    '''  The adjoint model for Single Pixel Light Field camera based on CLIP
    #  1 codes: codes used in SPC: [Na, Na, H, W, N_pattern], should be cuda device
    #  2 spc_data_in: the spc measurement, cuda-float, shaped in [u, v, Npattern],
    '''
    # [u, v, h, w, np] * [u, v, 1, 1, np]  --> [u, v, h, w, np] and then sum along np
    lf_recon = torch.einsum('uvhwp,uvhwp->uvhw', codes, spc_data_in.unsqueeze(2).unsqueeze(3) )
    return lf_recon #  [u, v, h, w]





#===========================exp model2============================

def A_sum_model(lf, psf_stack,device=None):
    if device is None:
        device = lf.device
    lf = lf.to(device)                  
    psf_stack = psf_stack.to(device)    
    h, w = lf.shape[-2:]
    num = psf_stack.shape[0]
    if_rgb = lf.dim() == 5

    frac = psf_stack.abs().sum(dim=(-2, -1)).clamp_min(1e-12)  # [num]
    weight = psf_stack / frac[:, None, None]                   # [num, nu, nv]

    if if_rgb:
        meas = torch.einsum('nuv,uvchw->nchw', weight, lf)
    else:
        meas = torch.einsum('nuv,uvhw->nhw', weight, lf)

    return meas

def AT_sum_model(meas, psf_stack, device=None):
    if device is None:
        device = meas.device

    meas = meas.to(device)
    psf_stack = psf_stack.to(device)
    

    if_rgb = meas.dim() == 4  

    frac = psf_stack.abs().sum(dim=(-2, -1)).clamp_min(1e-12)  
    weight = psf_stack / frac[:, None, None]                   

    if if_rgb:
        lf_back = torch.einsum('nuv,nchw->uvchw', weight, meas)
    else:
        lf_back = torch.einsum('nuv,nhw->uvhw', weight, meas)

    return lf_back




