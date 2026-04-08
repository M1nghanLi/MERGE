import os
import torch
import numpy as np
import time

from torch.optim import Adam, AdamW

from MERGE_model import *
from CreateData import *
from utils import *
from torchdeq.core import get_deq


from DenoisingLIB import *
#from MyDenoisersLib import load_denoiser


class DEQ_MERGE(nn.Module):
    def __init__(self, MERGE_MLP, depthestimator, denoiser, transfomation_param, lfshape, disp_max_value):
        super().__init__()
        self.MERGE_MLP = MERGE_MLP
        self.depthestimator = depthestimator
        self.denoiser = denoiser
        if isinstance(transfomation_param, torch.nn.Parameter):
            self.transfomation_param = transfomation_param
        else:
            self.transfomation_param = torch.nn.Parameter(transfomation_param)
        self.coo_im = coo_gen((lfshape[-2], lfshape[-1]))

        if len(lfshape)==4:
            H, W=lfshape[-2], lfshape[-1]
            self.bias_img = torch.nn.Parameter(torch.zeros(1, 1, H, W))
        elif len(lfshape)==5:
            H, W=lfshape[-2], lfshape[-1]
            self.bias_img = torch.nn.Parameter(torch.zeros(1, 3, H, W))
        else:
            raise ValueError('lfshape should be of length 4 or 5')
        self.disp_max_value = disp_max_value

    def forward(self, lf):
        u,v = lf.shape[:2]
        h,w = lf.shape[-2:]
        if lf.dim() == 4:
            guide_img = lf[u//2,v//2].clone().unsqueeze(0).unsqueeze(0)
            bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
            guide_img = guide_img + bias_on_device
            guide_img = guide_img.expand(1, 3, -1, -1)
        elif lf.dim() == 5:
            guide_img = lf[u//2,v//2].clone().unsqueeze(0)  # [1, 3, h, w]
            bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
            guide_img = guide_img + bias_on_device
            
        

        #=======================
        disparity_relative = self.depthestimator(guide_img).squeeze(0)/self.disp_max_value  # shape=(h,w)
        disparity_abs = torch.tanh(self.transfomation_param[0].to(disparity_relative.device) * disparity_relative 
                                   + self.transfomation_param[1].to(disparity_relative.device))
        lf_recon_raw, _ = self.MERGE_MLP(self.coo_im, disparity_abs)
        
        if lf.dim() == 4:
            lf_recon = einops.rearrange(lf_recon_raw.squeeze(-1), '(nu nv h w) -> nu nv h w', nu=u, h=h, w=w)
        else:
            lf_recon = einops.rearrange(lf_recon_raw, '(nu nv h w) c -> nu nv c h w', nu=u, nv=v, h=h, w=w, c=3)
        
        lf_recon = self.denoiser(lf_recon).to(lf.device)

        return lf_recon
    
    def forward_full(self, lf):
        with torch.no_grad():
            u,v = lf.shape[:2]
            h,w = lf.shape[-2:]
            if lf.dim() == 4:
                guide_img = lf[u//2,v//2].clone().unsqueeze(0).unsqueeze(0)
                bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
                guide_img = guide_img + bias_on_device
                guide_img = guide_img.expand(1, 3, -1, -1)
            elif lf.dim() == 5:
                guide_img = lf[u//2,v//2].clone().unsqueeze(0)  # [1, 3, h, w]
                bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
                guide_img = guide_img + bias_on_device
            
            disparity_relative = self.depthestimator(guide_img).squeeze(0)/self.disp_max_value  # shape=(h,w)
            disparity_abs = torch.tanh(self.transfomation_param[0].to(disparity_relative.device) * disparity_relative 
                                    + self.transfomation_param[1].to(disparity_relative.device))
            lf_recon_raw, disparity_allview_raw = self.MERGE_MLP(self.coo_im, disparity_abs)
            
            if lf.dim() == 4:
                lf_recon = einops.rearrange(lf_recon_raw.squeeze(-1), '(nu nv h w) -> nu nv h w', nu=u, h=h, w=w)
            else:
                lf_recon = einops.rearrange(lf_recon_raw, '(nu nv h w) c -> nu nv c h w', nu=u, nv=v, h=h, w=w, c=3)
            
            lf_recon = self.denoiser(lf_recon).to(lf.device)
            disparity_allview = einops.rearrange(disparity_allview_raw, '(nview h w) xy -> nview h w xy', h=h, w=w, xy=2)
        return lf_recon, disparity_allview


def return_MERGE_componets(config):

    lfshape                 =config['lfshape']
    depthmodelname          =config['depthmodelname']
    main_device             =config['main_device']
    depth_device            =config['depth_device']
    denoiser_device         =config['denoiser_device']
    omega0                  =config['MLPomega0']
    sigma0                  =config['MLPsigma0']
    hiddenlayers            =config['MLPhiddenlayers']
    hiddenfeatures          =config['MLPhiddenfeatures']

    if_rgb=(len(lfshape)==5)
    u,v=lfshape[0],lfshape[1]
    h,w=lfshape[-2],lfshape[-1]



    mp_devices = []
    try:
        if 'gpu_list' in config:
            for i in config['gpu_list']:
                mp_devices.append(f'cuda:{i}')
        else:
            n_gpus = torch.cuda.device_count()

            for i in range(2, 7):
                mp_devices.append(f'cuda:{i}')
        
        total_layers = 1 + hiddenlayers + 2  # first + hidden + (output+sigmoid)
        
        print(f"\n[GPU Strategy] Network: hidden_layers={hiddenlayers}, hidden_features={hiddenfeatures}")
        print(f"[GPU Strategy] Total MLP layers: {total_layers}")
        print(f"[GPU Strategy] Using all {len(mp_devices)} GPUs: {mp_devices}")
        print(f"[GPU Strategy] v2 strategy will assign remainder to LAST GPUs to avoid overloading GPU 3\n")
    except Exception as e:
        print(f"[GPU Strategy] Error: {e}")
        mp_devices = []

    if len(mp_devices) > 0 :
        if config['scale_alpha_mode'] =='learned':
            MERGE_MLP = COLF_Wire_rand_multigpu(  
                input_dim=2,
                hidden_features=hiddenfeatures,
                hidden_layers=hiddenlayers,
                out_features=3 if if_rgb else 1,
                N_view=u * v,
                hash_length=(h, w),
                omega_0=omega0,
                sigma_0=sigma0,
                need_split=True,  
                mirror=config.get('MLP_mirror', False),
                device=main_device,
                chunk_size=2**12,
                mp_devices=mp_devices
            )
        elif config['scale_alpha_mode'] =='finetune':
            MERGE_MLP = COLF_Wire_split_multigpu( 
                input_dim=2,
                hidden_features=hiddenfeatures,
                hidden_layers=hiddenlayers,
                out_features=3 if if_rgb else 1,
                N_view=u * v,
                hash_length=(h, w),
                omega_0=omega0,
                sigma_0=sigma0,
                need_split=True, 
                mirror=config.get('MLP_mirror', False),
                device=main_device,
                chunk_size=2**12,
                mp_devices=mp_devices
            )


    # 2. depth estimation model
    if depthmodelname=='depthanythingb':
        depthmodel = get_depthanything_model('vitb',device=depth_device)# vits--->vitb--->vitl
        depthestimator = lambda x:depthmodel.My_DepthAnything_forward(x)
    elif depthmodelname=='depthanythingl':
        depthmodel = get_depthanything_model('vitl',device=depth_device)# vits--->vitb--->vitl
        depthestimator = lambda x:depthmodel.My_DepthAnything_forward(x)
    elif depthmodelname=='depthanythings':
        depthmodel = get_depthanything_model('vits',device=depth_device)# vits--->vitb--->vitl
        depthestimator = lambda x:depthmodel.My_DepthAnything_forward(x)
    elif depthmodelname=='depthanything3gl':
        depthmodel = DepthAnything3Estimator(model_name='DA3NESTED-GIANT-LARGE', device=depth_device)  # DA3NESTED-GIANT-LARGE/DA3MONO-LARGE
        depthestimator = lambda x: depthmodel(x)
    elif depthmodelname=='depthanything3l':
        depthmodel = DepthAnything3Estimator(model_name='DA3MONO-LARGE', device=depth_device)  # DA3NESTED-GIANT-LARGE/DA3MONO-LARGE
        depthestimator = lambda x: depthmodel(x)
    elif depthmodelname=='zoedepth':
        depthmodel = ZoeDepthEstimator(device=depth_device)
        depthestimator = lambda x: 1 / depthmodel(x)
    elif depthmodelname=='vggt':
        depthmodel = VGGTEstimator(device=depth_device)
        depthestimator = lambda x: 1 / depthmodel(x)
    else:
        raise ValueError('Unsupported depth model name')
    


    # 3. denoiser
    denoise_net = load_denoiser(config)

    if not if_rgb:
        denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv h w -> (nu nv) h w')), '(nu nv) h w -> nu nv h w', nu=u)
    else:
        #denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv c h w -> (nu nv c) h w')), '(nu nv c) h w -> nu nv c h w', nu=u,c=3)
        denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv c h w -> (nu nv) c h w')), '(nu nv) c h w -> nu nv c h w', nu=u,c=3)
    

    
    #transfomation_param
    transfomation_param = torch.ones(2, dtype=torch.float32, device=main_device) * 1e-2
    transfomation_param.requires_grad_(True)

    return MERGE_MLP, depthestimator, denoiser, transfomation_param , denoise_net

def train_model(A,meas_data,config):
    main_device = config['main_device']
    depth_device = config['depth_device']
    denoiser_device = config['denoiser_device']
    lfshape = config['lfshape']
    nu,nv=lfshape[0],lfshape[1]
    #H,W=meas_data.shape[-2],meas_data.shape[-1]\
    H,W=lfshape[-2],lfshape[-1]
    lr = config['lr']
    
    meas_data = meas_data.to(main_device)
    
    coo_im = coo_gen((H, W)).to(main_device)
    components = return_MERGE_componets(config)
    MERGE_MLP, depthestimator, denoiser, transfomation_param, denoise_net=components

    start_t = time.perf_counter()
    loss_func = nn.L1Loss()
    if_rgb=(len(lfshape)==5)
    optimizer = AdamW(MERGE_MLP.mlp_lf.parameters(), lr=lr)
#=========================S1 coarse training========================
    for iter in range(config['S1_iter']):
        im_raw = MERGE_MLP.mlp_lf(coo_im)

        if not if_rgb:
            im_recon = einops.rearrange(im_raw.squeeze(-1), '(h w) -> h w', h=H, w=W)
            im_recon = denoise_net(im_recon.unsqueeze(0)).squeeze()
            lf_warm = im_recon.expand(nu, nv, H, W)
        else:
            im_recon = einops.rearrange(im_raw, '(h w) c -> c h w ', h=H, w=W, c=3)
            im_recon = denoise_net(im_recon.unsqueeze(0)).squeeze(0)
            lf_warm = im_recon.expand(nu, nv, 3, H, W)
        meas_hat = A(lf_warm.to(main_device, non_blocking=True))
        loss = loss_func(meas_hat, meas_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (np.mod(iter, 20) == 0):
            print('Iter: {0:.1f} loss: {1:.5f} '.format(iter, loss.item()))



#==========================S2 setup=========================
    lf_input = lf_warm.detach().clone()
    init_disparity = depthestimator(norm(im_recon))
    disp_max_value = torch.max(torch.abs(init_disparity)).item()
    DEQ_MERGEmodel = DEQ_MERGE(MERGE_MLP, depthestimator, denoiser, transfomation_param, lfshape, disp_max_value).to(main_device)
    deq = get_deq(core='sliced', ift=True)
    

    optimizer = AdamW(DEQ_MERGEmodel.parameters(), lr=lr)

    DEQ_cycle = config.get('DEQ_cycle', 1)
    
    dir_path = os.path.join(config['save_dir'],'rawdata')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    if config.get('bias_start_iter', 0)!=0:
        param_groups = [
            {'params': [p for n, p in DEQ_MERGEmodel.named_parameters() if 'bias_img' not in n], 'lr': lr},
            {'params': [p for n, p in DEQ_MERGEmodel.named_parameters() if 'bias_img' in n], 'lr': 0.0}
        ]
        optimizer = AdamW(param_groups)
        delayed_bias_training = True
    else:
        # Standard: optimize all parameters together
        optimizer = AdamW(DEQ_MERGEmodel.parameters(), lr=lr)
        delayed_bias_training = False

#==========================S2 DEQ training=========================
    s2_records = []
    s2_loss_list = []
    for iter in range(config['S2_iter']):
        if delayed_bias_training and iter == config['bias_start_iter']:
            print(f"\n[Iter {iter}] Activating bias_img training (setting lr from 0 to {lr})")
            for param_group in optimizer.param_groups:
                if any('bias_img' in n for n, p in DEQ_MERGEmodel.named_parameters() 
                       if any(p is param for param in param_group['params'])):
                    param_group['lr'] = lr



        #if (iter % DEQ_cycle == 0)  and (iter >= config['DEQstart_iter'] and (iter<config['S2_iter']-50)):
        if (iter % DEQ_cycle == 0)  and (iter >= config['DEQstart_iter']):
            torch.cuda.empty_cache()
            z_out, _ = deq(DEQ_MERGEmodel, lf_input,
                            solver_kwargs=dict(f_solver='anderson'),
                            stop_mode='abs')
            lf_star = z_out[-1]
            with torch.no_grad():
                residual = torch.mean(torch.abs((lf_star-DEQ_MERGEmodel(lf_star))))
                #lf_input = lf_star.detach().clone()
                
        else:
            lf_star = DEQ_MERGEmodel(lf_input)
            with torch.no_grad():
                #residual = torch.mean(torch.abs((lf_star-lf_input)))
                residual = torch.mean(torch.abs((lf_star-DEQ_MERGEmodel(lf_star))))
        
        meas_hat = A(lf_star.to(main_device, non_blocking=True))



        loss = loss_func(meas_hat, meas_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lf_input = lf_star.detach().clone()

        loss_value = float(loss.item())
        residual_value = float(residual.item())
        s2_loss_list.append(loss_value)
        s2_records.append({
            'iter': int(iter),
            'loss': loss_value,
            'residual': residual_value,
        })

        print(f'Iter:{iter}/{config["S2_iter"]}, loss:{loss.item():.5f} residual:{residual.item():.2e}')

    stop_t = time.perf_counter()
    print('total_train_time: {}s'.format(stop_t - start_t))
    with torch.no_grad():
        final_lf, final_disparity_full = DEQ_MERGEmodel.forward_full(lf_star)
    k = MERGE_MLP.compute_scale_alpha()
    print(k)

    return final_lf.detach(), final_disparity_full.detach(), DEQ_MERGEmodel #shape=(nu,nv,h,w), (nview,h,w,2)




if __name__ == '__main__':

    save_dir = f'CLIP_output'


    gpu_id = 0
    device = f'cuda:{gpu_id}'
    codes,meas_data,A,AT,lfresolution,lf_data = get_SinglePixelImagingMeasurement(scene=f'scene{1}_LF_Data_SPC.mat', device=device)
    
    #meas_data = (meas_data + torch.randn_like(meas_data)*noise_level/2).clamp_(min=0)

    lfshape=lf_data.shape
    u,v,h,w=lfshape
#============ MERGE Reconstruct ==============
    print(f'Start MERGE Reconstruct...')
    config = {}

    config['save_dir']          = save_dir
    config['lfshape']           = lfshape
    config['lr']                = 5e-3
    config['S1_iter']           = 400
    config['S2_iter']           = 200   
    config['gpu_list']          = [gpu_id]
    config['main_device']       = config['gpu_list'][0]
    config['depth_device']      = config['gpu_list'][0]
    config['denoiser_device']   = config['gpu_list'][0]
    config['depthmodelname']    = 'depthanythings'  
    # 'depthanythingb'/'depthanythingl'/'depthanythings'/'zoedepth'/'vggt' / 'depthanything3gl'/'depthanything3l'/'depthpro'
    config['denoiser']          = 'ffdnet'
    config['lambda']            = 6e-2  #6e-2
    config['MLPhiddenlayers']   = 3
    config['MLPhiddenfeatures'] = 256  
    config['MLPomega0']         = 5.0  
    config['MLPsigma0']         = 5.0  
    config['need_iter_disp']    = False  
    config['delay_bias']        = True  
    config['bias_start_iter']   = 80
    config['DEQ_cycle']         = 1
    config['DEQstart_iter']     = 800  
    config['scale_alpha_mode']  = 'learned' # 'finetune' or 'learned'
    #with open(os.devnull, 'w') as fnull: 
        #with contextlib.redirect_stdout(fnull): 
    lf_merge, final_disparity_full, DEQ_MERGEmodel = train_model(A,meas_data.to(config['main_device']),config)
    
    #MERGE data save
    #1,raw data
    disparity_path = os.path.join(save_dir,'rawdata',f'disparity.pt')
    dir_path = os.path.dirname(disparity_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    lf_path = os.path.join(save_dir,'rawdata',f'merge_lf.pt')
    torch.save(lf_merge,lf_path)
    torch.save(final_disparity_full,disparity_path)
    #2,disparity img
    k = DEQ_MERGEmodel.MERGE_MLP.compute_scale_alpha()
    finaldisparity = disp_allview_to_disparity(final_disparity_full)



    save_img(lf_merge[u//2,v//2], os.path.join(save_dir,f'sub_merge.png'))
    #save_depthimg(finaldisparity,os.path.join(save_dir,f'disparity_num{num}.png'))
    Plot(finaldisparity.cpu().detach().squeeze(), cmap='RdBu',savepath=os.path.join(save_dir,f'disparity.png'))
    light_field_to_gif(norm(lf_merge), os.path.join(save_dir,f'lfgif.gif'))
    light_field_to_video_hq(norm(lf_merge), os.path.join(save_dir,f'lfvideo.mp4'))

    psnr_merge = psnr_torch(norm(lf_merge),norm(lf_data), 1.0)
    print(f'PSNR (MERGE): {psnr_merge:.2f} dB')

