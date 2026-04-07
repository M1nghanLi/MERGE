# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:36:44 2024

@author: Xiaohua Feng
"""
import torch
from vision_networks import *
import einops
from einops import rearrange

def load_denoiser(opt):
    # Load a denoising network specificed in opt
    # Author: Xiaohua Feng
    reg_lambda = opt['lambda']
    #device = opt.get('device', None)
    device = opt.get('device', opt.get('denoiser_device', None))
    if (opt['denoiser'] == 'ffdnet'):
        print(f"[DenoisingLIB] Loading ffdnet (grayscale)")
        network = load_model(device=device)
        denoise = lambda noise_x: denoise_net(network, opt, noise_x, reg_lambda)
    elif (opt['denoiser'] == 'ffdnet_rgb'):
        print(f"[DenoisingLIB] Loading ffdnet_rgb")
        network = load_model_ffdnetrgb(device=device)
        denoise = lambda noise_x: denoise_net(network, opt, noise_x, reg_lambda)
    elif (opt['denoiser'] == 'proxtv'):
        denoise = lambda noise_x: denoise_ProxTV3D(noise_x, reg_lambda)
    # elif(opt['denoiser'] == 'proxl1'):
    #     denoise = lambda noise_x: proxl1(noise_x, reg_lambda)
    elif(opt['denoiser'] == 'fastdvd'):
        network = load_fastdvdnet()
        denoise = lambda noise_x: denoise_fastdvdnet(network, noise_x, reg_lambda)
    elif(opt['denoiser'] == 'rvrt'):
        network = loadRVRT_model()
        denoise = lambda noise_x: denoise_rvrt(network, noise_x, reg_lambda)
    else:
        if(opt['denoiser'] == 'drunet'):
            network = load_model_DRUNet()
        elif(opt['denoiser'] == 'drunet_rgb'):
            network = load_model_DRUNet_rgb(device=device)
        elif(opt['denoiser'] == 'restormer'):
            network = load_model_Restormer()
        elif(opt['denoiser'] == 'kbnet'):
            network = load_model_KBNet()
        elif(opt['denoiser'] == 'swinir'):
            network = load_model_swinIR()
        elif opt['denoiser'] == 'fdncnn':
            network = load_model_FDnCNN()

        else: 
            network = load_model()

        denoise = lambda noise_x : denoise_net(network, opt, noise_x, reg_lambda)
                
    return denoise

def denoise_net(net, opt, noisy, sigma_hat):
    # For using with the output of inr models, with range of [-1, 1] scaling to the range of [0 1] is
    # done by adding 1 and divide by 2
    
    # 特殊处理 ffdnet_rgb: 支持 [b, 3, h, w] 输入
    if opt['denoiser'] in ['ffdnet_rgb', 'drunet_rgb']:
        # 检查输入维度
        if noisy.dim() == 4:
            # [b, 3, h, w] 输入
            Nz, C, Nx, Ny = noisy.shape
            is_rgb_batch = True
        elif noisy.dim() == 3:
            # 旧格式 [Nz, Nx, Ny] - 不应该用于RGB去噪器，但保持兼容
            Nz, Nx, Ny = noisy.shape
            is_rgb_batch = False
        else:
            raise ValueError(f"Unsupported noisy shape for {opt['denoiser']}: {noisy.shape}")
    else:
        # 其他去噪器保持原有逻辑
        Nz, Nx, Ny = noisy.shape
        is_rgb_batch = False

    # ========padding =============
    h, w = noisy.shape[-2:]
    img_multiple_of = 8
    if h%img_multiple_of!=0 or w%img_multiple_of!=0:
        H = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        noisy = nn.functional.pad(noisy, (0, padw, 0, padh), mode='reflect')
    
    # 更新填充后的尺寸
    if opt['denoiser'] in ['ffdnet_rgb', 'drunet_rgb'] and is_rgb_batch:
        Nz, C, Nx, Ny = noisy.shape
    else:
        Nz, Nx, Ny = noisy.shape

    # 确保噪声与网络在同一设备/数据类型，然后再做归一化计算。
    # 备注：训练中可能启用 AMP(bf16/fp16)。FFDNet 等网络参数通常为 float32，
    # 若直接传入 bf16 会触发 conv2d 的 input/bias dtype 不一致错误。
    if isinstance(net, torch.nn.Module):
        param = next(net.parameters())
        target_device = param.device
        target_dtype = param.dtype
    else:
        target_device = noisy.device
        target_dtype = noisy.dtype
    noisy = noisy.to(device=target_device, dtype=target_dtype)
    # 归一化到 [0,1]
    min_val = torch.min(noisy)
    noisy3D = noisy - min_val
    max_val = torch.max(noisy3D)
    noisy3D = noisy3D / max_val
    img = torch.zeros_like(noisy3D, device=target_device)
    
    if(opt['denoiser'] == 'ffdnet'):
        sigma = torch.full((1,1,1,1), sigma_hat, device=target_device, dtype=noisy3D.dtype)  # noise power
        for K in range(noisy3D.shape[0]):
            img[K,:,:] = net(noisy3D[K,:,:].unsqueeze(0).unsqueeze(0), sigma).squeeze(0).squeeze(0)
    
    elif(opt['denoiser'] == 'ffdnet_rgb'):
        # FFDNet RGB: 支持批量 [b, 3, h, w] 处理
        if is_rgb_batch:
            sigma = torch.full((Nz, 1, 1, 1), sigma_hat, device=target_device, dtype=noisy3D.dtype)
            img = net(noisy3D, sigma)
        else:
            # 保持旧的逐帧处理逻辑（虽然不推荐用于RGB）
            sigma = torch.full((1,1,1,1), sigma_hat, device=target_device, dtype=noisy3D.dtype)
            for K in range(noisy3D.shape[0]):
                img[K,:,:] = net(noisy3D[K,:,:].unsqueeze(0).unsqueeze(0), sigma).squeeze(0).squeeze(0)
            
    elif(opt['denoiser'] == 'drunet'):
        noise_map = torch.ones([1,1,Nx,Ny], dtype=torch.float32, device=target_device) * sigma_hat
        for K in range(noisy3D.shape[0]):
            img[K,:,:] = net(torch.cat([noisy3D[K,:,:].unsqueeze(0).unsqueeze(0), noise_map], dim =1) ).squeeze(0).squeeze(0)
    elif(opt['denoiser'] == 'drunet_rgb'):
        if is_rgb_batch:
            noise_map = torch.ones([1, 1, Nx, Ny], dtype=torch.float32, device=target_device) * sigma_hat
            for K in range(Nz):
                img[K] = net(torch.cat([noisy3D[K].unsqueeze(0), noise_map], dim=1)).squeeze(0)
        else:
            raise ValueError(f"Input to {opt['denoiser']} must be rgb 4D tensor [b, 3, h, w]")
    elif (opt['denoiser'] == 'fdncnn'):
        noise_map = torch.ones([1, 1, Nx, Ny], dtype=torch.float32,device=noisy3D.device) * sigma_hat
        for K in range(noisy3D.shape[0]):
            img[K, :, :] = net(torch.cat([noisy3D[K, :, :].unsqueeze(0).unsqueeze(0),noise_map], dim=1)).squeeze(0).squeeze(0)
    elif (opt['denoiser'] == 'ircnn'):
        # 新增 IRCNN 支持：构造 sigma 张量并传递
        sigma = torch.full((1,1,1,1), sigma_hat).type_as(noisy3D)
        for K in range(noisy3D.shape[0]):
            img[K,:,:] = net(noisy3D[K,:,:].unsqueeze(0).unsqueeze(0), sigma).squeeze(0).squeeze(0)
    #elif (opt['denoiser'] == 'restormer_ming'):

    else:
        ''' This case can deal with restormer (~10 times slower), KBNet (~10 times slower), swinir (~100 times slower than drunet)
            Refers to the RED by fixed point projection paper (RED-PRO) for using denoisiers with fixed sigma
        '''
        # with torch.no_grad():
        #     for K in range(img.shape[0]):
        #         denoised_tmp = net(noisy3D[K,:,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        #         img[K,:,:] = denoised_tmp * sigma_hat + noisy3D[K,:,:] * (1-sigma_hat)
        for K in range(img.shape[0]):
            #img[K,:,:] = net(noisy3D[K,:,:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            denoised_tmp = net(noisy3D[K, :, :].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            img[K, :, :] = denoised_tmp * sigma_hat + noisy3D[K, :, :] * (1 - sigma_hat)
                
    img = img*max_val + min_val
    #img = img * 2- 1
    
    # 裁剪回原始尺寸
    if opt['denoiser'] in ['ffdnet_rgb', 'drunet_rgb'] and is_rgb_batch:
        img = img[:, :, :h, :w]
    else:
        img = img[:, :h, :w]
    
    return img

# **********************************************************************
''' Using ProxTV (3D or a single z) or ProxHessian for denoising a volumetric image
'''
# **********************************************************************
# **********************************************************************
def tv3d_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (C, H, W) holding an input image.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
    """
    w_variance = torch.sum(torch.pow(img[:,:,:-1] - img[:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:-1,:] - img[:,1:,:], 2))
    # z_variance = torch.sum(torch.pow(img[:-1,:,:] - img[1:,:,:], 2))
    loss = torch.sqrt(h_variance + w_variance ) # + z_variance
    return loss


class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image, reg_param):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        # self.regularization_term = tv_loss()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image
        self.reg_param = reg_param
        
    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + self.reg_param * tv3d_loss(self.clean_image) 

    def get_clean_image(self):
        return self.clean_image
 
def denoise_ProxTV3D(noisy_image, sigma_hat):
    # define the total variation denoising network
    # noisy_image input is in shape of Nx, Ny, Nz
    tv_denoiser = TVDenoise(noisy_image, sigma_hat)
    
    # define the optimizer to optimize the 1 parameter of tv_denoiser
    # optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr = 0.01, momentum=0.9)
    optimizer = torch.optim.Adam(tv_denoiser.parameters(), lr= 1e-2)
    num_iters = 300
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = tv_denoiser()
        # if i % 50 == 0:
        #     print("Loss in iteration {} of {}: {:.6f}".format(i, num_iters, loss.item()))
        loss.backward()
        optimizer.step()
    # print("TV Loss: {:.6f}".format(loss.item()))

    img_clean = tv_denoiser.get_clean_image().detach()
    return img_clean

# **********************************************************************
''' Using fastDVDNet for denoising a volumetric image
'''
# **********************************************************************
# **********************************************************************

NUM_IN_FR_EXT = 5 # temporal size of patch for fastDVDnet

def load_fastdvdnet():
    from models.fastdvd_models import FastDVDnet
    model_path = os.getcwd() + '/model_zoo/fastdvdnet_model.pth'
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
    device = torch.device('cuda')

    state_temp_dict = torch.load(model_path, map_location=device)
    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(state_temp_dict)
    
   	# Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()  
    return model

def denoise_fastdvdnet(model, vol_in, noise_std):
    r"""Denoises a volumetric image with FastDVDnet.
    Note that the input volmetric image should be multiple of 4 in H and W dim.
	Args:
		seq: Tensor. [Nz Nx Ny] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the video noise level
		Nz_f: FOV along the z dimension
		model: instance of the FastDVDNet
	Returns:
		clean_vol: Tensor, [Nz Nx Ny]
    """
    # ========padding =============
    _,h,w =vol_in.shape
    img_multiple_of = 8
    if h%img_multiple_of!=0 or w%img_multiple_of!=0:
        H = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        vol_in = nn.functional.pad(vol_in, (0, padw, 0, padh), mode='reflect')




    vol_in = vol_in.unsqueeze(1) # [nz 1 nx ny]
    
    min_val = torch.min(vol_in)
    vol_in = vol_in - min_val
    max_val = torch.max(vol_in)
    vol_in = vol_in / max_val
    
    Nz_f = 3*(NUM_IN_FR_EXT + 0)
    # init arrays to handle contiguous frames and related patches
    Nz, C, H, W = vol_in.shape
    ctrlfr_idx = int((Nz_f-1)//2)
    inframes = list()
    clean_vol = torch.empty((Nz, 1, H, W)).to(vol_in.device)

    # build noise map from noise std---assuming Gaussian noise
    noise_map = torch.ones((H, W), dtype=torch.float32, device = 'cuda') * noise_std
    noise_map = noise_map.expand((1, 1, H, W))
    for fridx in range(Nz):
        # load input frames
        if not inframes:
		    # if list not yet created, fill it with temp_patchsz frames
            for idx in range(Nz_f):
                relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
                inframes.append(vol_in[relidx])
        else:
            del inframes[0]
            relidx = min(fridx + ctrlfr_idx, -fridx + 2*(Nz-1)-ctrlfr_idx) # handle border conditions
            inframes.append(vol_in[relidx])

        inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, Nz_f, H, W)).to(vol_in.device)
        # append result to output list
        # print(inframes_t.shape)
        #with torch.no_grad():
        clean_vol[fridx] = model(inframes_t, noise_map)[:,0,:,:] *max_val + min_val
    clean_vol = clean_vol[:, :,:h, :w]
	# free memory up
    del inframes
    del inframes_t
    return clean_vol.squeeze(1)

''' 
Adopting Recurrent video denoising (RVRT) for regularization in inverse problems
code changed from the original github
Credits: https://github.com/JingyunLiang/RVRT
'''
def loadRVRT_model():
    ''' prepare model'''
    from models.network_rvrt import RVRT as net
    # define model
    model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12, attention_heads=12,
                attention_window=[3, 3], nonblind_denoising=True, cpu_cache_length=100)

    # load model
    model_path = os.getcwd() +'/model_zoo/RVRT_videodenoising_DAVIS_16frames.pth'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval().cuda()
    return model

def test_video(lq, model, args):
        '''LFdata_RAW the video as a whole or as clips (divided temporally). '''
        num_frame_testing = args['tile'][0]
        if num_frame_testing:
            # LFdata_RAW as multiple clips if out-of-memory
            num_frame_overlapping = args['tile_overlap'][0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h, w, device='cuda')
            W = torch.zeros(b, d, 1, 1, 1, device='cuda')

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, args)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1), device='cuda')

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # LFdata_RAW as one clip (the whole video) if you have enough memory
            window_size = [2,8,8]
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' LFdata_RAW the clip as a whole or as patches. '''

    window_size = [2,8,8]
    size_patch_testing = args['tile'][1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args['tile_overlap'][1]
        not_overlap_border = True

        # LFdata_RAW patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = torch.zeros(b, d, c, h, w)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch)
                W[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach()

        output = output[:, :, :, :h_old, :w_old]

    return output

def denoise_rvrt(model, vol, sigma_hat):
    # RVRT specifics: input vol should be shaped into [nz, nx, ny], nz should be divisible by 3
    min_val = torch.min(vol)
    vol = vol - min_val
    max_val = torch.max(vol)
    vol = vol / max_val
    
    nz, nx, ny = vol.shape
    args = {}
    args['tile'] = [0, 0, 0]        # default=[100,128,128], info: Tile size, [0,0,0] for no tile during testing (testing as a whole)
    args['tile_overlap'] =[2,20,20] # help='Overlapping of different tiles'
    vol = rearrange(vol, '(nt nb) nx ny->nt nb nx ny', nb=3).unsqueeze(0)  # shaped into [1, Nz, 3, Nx, Ny]
    sigma = torch.ones([1,vol.shape[1],1,nx, ny], dtype=torch.float32, device = 'cuda') * sigma_hat
    vol = torch.cat([vol, sigma], dim = 2)
    # inference
    with torch.no_grad():
        vol_c = test_video(vol, model, args)
    vol_c = vol_c[0,:,:,:,:]
    vol_c = rearrange(vol_c, 'nt nb nx ny->(nt nb) nx ny', nb=3)
    vol_c = vol_c.cuda() * max_val + min_val

    return vol_c



def load_model_FDnCNN():
    """
    加载 FDnCNN 预训练模型
    参数 opt 可额外提供:
        opt['fdncnn_model'] : 'fdncnn_gray' (默认) /
                              'fdncnn_color' 等
    """
    import os, torch
    from models.network_dncnn import FDnCNN as net

    n_channels  = 1
    model_path = os.getcwd() + '/model_zoo/fdncnn_gray.pth'
    model = net(in_nc=n_channels+1, out_nc=n_channels,
                nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path),
                          strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.cuda() if torch.cuda.is_available() else model