"""
Created on 2025/6/4 15:18

@author: Minghan Li
we have 3 forward model: CSLIP CLIP LIFT, so we use each forward model to create simulation data
"""
import einops
import torch

from Forward_Model import *
from MERGE_model import *
import hdf5storage
import os
from utils import norm

#===========================CSLIP model============================

def get_CSLIPMeasurement(path,num,angle_dec=0,spatial_dec=0, device='cuda'):
    lf_data = torch.load(path, weights_only=True).to(device)
    #lf_data = lf_data / torch.max(torch.abs(lf_data)) * 0.5
    lf_data_pnp = lf_data / torch.max(torch.abs(lf_data))
    #lf_data = lf_data_pnp*0.5

    # decimate the measurement data
    if not angle_dec==0:
        lf_data_pnp = lf_data[::angle_dec, ::angle_dec, :, :]
    if not spatial_dec == 0:
        if len(lf_data_pnp.shape)==4:
            lf_data_pnp = F.interpolate(lf_data_pnp, scale_factor=1 / spatial_dec, mode='bicubic',antialias=True)
        elif len(lf_data_pnp.shape)==5:
            u=lf_data_pnp.shape[0]
            v=lf_data_pnp.shape[1]
            lf_data_pnp = einops.rearrange(lf_data_pnp, 'u v c h w -> (u v) c h w')
            lf_data_pnp = F.interpolate(lf_data_pnp, scale_factor=1 / spatial_dec, mode='bicubic',antialias=True)
            lf_data_pnp = einops.rearrange(lf_data_pnp, '(u v) c h w -> u v c h w',u=u,v=v)

    lf_data_pnp = lf_data_pnp / torch.max(torch.abs(lf_data_pnp))
    lf_data = lf_data_pnp * 0.5
    
    if len(lf_data.shape) == 4:
        u, v, h, w = lf_data.shape
    elif len(lf_data.shape) == 5:
        u, v, c, h, w = lf_data.shape

    # define the linear model
    codes = torch.randint(0, 2, (num, h, w), device=device).float()
    A = lambda x: H_ForwardOperator_opt(codes, x, batch_size=50, usecheckpoint=False)
    #A = lambda x:x
    AT = lambda y: H_AdjointOperator_opt(y, codes, u, v, batch_size=50) #10
    meas_data = A(lf_data)
    meas_data_pnp = A(lf_data_pnp)

    # 返回正确的分辨率信息
    if len(lf_data.shape) == 4:
        lfresolution = [u, v, h, w]
    elif len(lf_data.shape) == 5:
        lfresolution = [u, v, c, h, w]

    #return model, meas_data, A, AT, [u,v,h,w], lf_data, lf_data_pnp, meas_data_pnp
    return meas_data, A, AT, lfresolution, lf_data, meas_data_pnp, lf_data_pnp


#===========================CLIP model============================
def get_SinglePixelImagingMeasurement(scene='scene6_LF_Data_SPC.mat',device='cuda'):
#read data
    data_dir = 'ExpData_total/CLIP/'
    mat_content = hdf5storage.loadmat(data_dir + scene)  # CLIP_Data_SPC
    codes = mat_content['codes']  # [h, w, u, v, n_meas]
    meas_data = mat_content['meas_data']  # [u v n_meas]
    lf_data = mat_content['LF_sim']  # [u v h w]

    h, w, u, v, n = codes.shape
    codes = einops.rearrange(numpy2cuda(codes), 'h w u v n-> (h w) (u v n)')

    # 2: preprocess the data
    for K in range(codes.shape[0]):
        codes[K, :] = codes[K, :] - torch.mean(codes[K, :])

    fac = 1.0 / torch.mean(torch.linalg.vector_norm(codes, ord=2, dim=1))
    codes = einops.rearrange(codes * fac, '(h w) (u v n)-> u v h w n', h=h, u=u, n=n).to(device)

    meas_data = numpy2cuda(meas_data)
    meas_data = meas_data - torch.mean(meas_data)
    meas_data = meas_data / torch.max(torch.abs(meas_data)) * 0.5  # [u v n_meas]

    # %% FSITA_PnP recon
    recon_shape = codes.shape[0:-1]
    A_model = CLIP_SPC(codes).to(device)
    #A = lambda x: CLIP_SP_Forward(codes, x)
    AT = lambda x: CLIP_SP_Adjoint(codes, x)

    lfresolution=[recon_shape[0],recon_shape[1],h,w]



    return codes,meas_data.to(device),A_model,AT,lfresolution,torch.from_numpy(lf_data).to(device)


#===========================Coded Aperture============================
def load_exp_data(num,meas_dir,pattern_path='ExpData_total/CodedAperture/modulation_ndarray.npy',
                  spatial_dec = None,if_rgb=False):
    

    clbt = imagefile2tensor(os.path.join(meas_dir,'1.bmp'),if_rgb=if_rgb).squeeze()
    h,w=clbt.shape[-2:]

    if not if_rgb:
        meas_data = torch.zeros([num,h,w])
    else:
        meas_data = torch.zeros([num,3,h,w])
    for i in range(num):
        img = imagefile2tensor(os.path.join(meas_dir,f'{i+1}.bmp'),normalize=False,if_rgb=if_rgb).squeeze()
        if if_rgb and img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        meas_data[i] = img.clone() 
    

    if spatial_dec is not None and spatial_dec != 0:
        if if_rgb:
            meas_data = F.interpolate(meas_data, scale_factor=1 / spatial_dec, mode='bicubic', antialias=True)
        else:
            meas_data = F.interpolate(meas_data.unsqueeze(1), scale_factor=1 / spatial_dec, mode='bicubic', antialias=True).squeeze(1)


    meas_data = norm(meas_data) * 0.5
    h,w = meas_data.shape[-2:]

    modulation_code = torch.from_numpy(np.load(pattern_path)).float().cuda()
    modulation_code = modulation_code[:num]
    modulation_code = torch.flip(modulation_code, dims=[-2])  

    A = lambda lf:A_sum_model(lf, modulation_code)
    AT = lambda y: AT_sum_model(y, modulation_code)

    return meas_data,A,AT





