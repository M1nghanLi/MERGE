import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim import Adam, AdamW
from MERGE_model import *
from CreateData import *
from utils import *


from torchdeq.core import get_deq
from torchdeq.solver import get_solver 



from DenoisingLIB import *


def _forward_in_chunks(net: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Forward a (potentially huge) 2D coordinate batch through `net` in chunks.

    This is used to avoid single-GPU OOM when evaluating `mlp_lf` on full-resolution
    coordinates (e.g. 1400x1400). It does NOT use checkpointing and does NOT change
    the configured chunk size; it simply iterates.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    outs = []
    n = x.shape[0]
    for i in range(0, n, chunk_size):
        outs.append(net(x[i : i + chunk_size]))
    return torch.cat(outs, dim=0)

def _freeze_depthmodel(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
class DEQ_MURDGE(nn.Module):
    def __init__(self, MURDGE_MLP, depthestimator, denoiser, transfomation_param, lfshape, disp_max_value):
        super().__init__()
        # register submodules / parameters
        self.MURDGE_MLP = MURDGE_MLP
        self.depthestimator = depthestimator
        self.denoiser = denoiser
        # ensure transformation params are registered as nn.Parameter so they show up in .parameters()
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

        lf_recon_raw, _ = self.MURDGE_MLP(self.coo_im, disparity_abs)
        
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

            lf_recon_raw, disparity_allview_raw = self.MURDGE_MLP(self.coo_im, disparity_abs)
            
            if lf.dim() == 4:
                lf_recon = einops.rearrange(lf_recon_raw.squeeze(-1), '(nu nv h w) -> nu nv h w', nu=u, h=h, w=w)
            else:
                lf_recon = einops.rearrange(lf_recon_raw, '(nu nv h w) c -> nu nv c h w', nu=u, nv=v, h=h, w=w, c=3)
            
            lf_recon = self.denoiser(lf_recon).to(lf.device)
            disparity_allview = einops.rearrange(disparity_allview_raw, '(nview h w) xy -> nview h w xy', h=h, w=w, xy=2)
        return lf_recon, disparity_allview
    def compute_base_disparity(self,lf):
        with torch.no_grad():
            u,v = lf.shape[:2]
            h,w = lf.shape[-2:]
            if lf.dim() == 4:
                guide_img = lf[u//2,v//2].clone().unsqueeze(0).unsqueeze(0)
                # Move bias to guide_img's device (for consistency with forward)
                bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
                guide_img = guide_img + bias_on_device
                guide_img = guide_img.expand(1, 3, -1, -1)
            elif lf.dim() == 5:
                guide_img = lf[u//2,v//2].clone().unsqueeze(0)  # [1, 3, h, w]
                # Move bias to guide_img's device (for consistency with forward)
                bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
                guide_img = guide_img + bias_on_device
            disparity_relative = self.depthestimator(guide_img).squeeze(0)/self.disp_max_value  # shape=(h,w)
            disparity_abs = torch.tanh(self.transfomation_param[0].to(disparity_relative.device) * disparity_relative 
                                    + self.transfomation_param[1].to(disparity_relative.device))
        return disparity_abs

def return_MURDGE_componets(config):

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

        # De-duplicate while preserving order
        mp_devices = list(dict.fromkeys(mp_devices))
        
        total_layers = 1 + hiddenlayers + 2  # first + hidden + (output+sigmoid)
        
        print(f"\n[GPU Strategy] Network: hidden_layers={hiddenlayers}, hidden_features={hiddenfeatures}")
        print(f"[GPU Strategy] Total MLP layers: {total_layers}")
        print(f"[GPU Strategy] Final MLP GPUs: {len(mp_devices)} -> {mp_devices}")
        print(f"[GPU Strategy] v2 strategy will assign remainder to LAST GPUs to avoid overloading GPU 3\n")
    except Exception as e:
        print(f"[GPU Strategy] Error: {e}")
        mp_devices = []

    if len(mp_devices) > 0 :
        if config['scale_alpha_mode'] =='learned':
            MURDGE_MLP = COLF_Wire_rand_multigpu(  
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
            MURDGE_MLP = COLF_Wire_split_multigpu( 
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
                chunk_size=config.get('mlp_chunk_size', 2**17),
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
    
    # Freeze depth model parameters to save memory
    if 'depthmodel' in locals() and depthmodel is not None:
        if hasattr(depthmodel, 'model'):
             _freeze_depthmodel(depthmodel.model)
        elif isinstance(depthmodel, torch.nn.Module):
             _freeze_depthmodel(depthmodel)


    # 3. denoiser
    denoise_net = load_denoiser(config)
    # Freeze denoiser parameters to save memory
    if denoise_net is not None and isinstance(denoise_net, torch.nn.Module):
        _freeze_depthmodel(denoise_net)

    if not if_rgb:
        denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv h w -> (nu nv) h w')), '(nu nv) h w -> nu nv h w', nu=u)
    else:
        #denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv c h w -> (nu nv c) h w')), '(nu nv c) h w -> nu nv c h w', nu=u,c=3)
        denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv c h w -> (nu nv) c h w')), '(nu nv) c h w -> nu nv c h w', nu=u,c=3)
    

    
    #transfomation_param
    transfomation_param = torch.ones(2, dtype=torch.float32, device=main_device) * 1e-2
    transfomation_param.requires_grad_(True)

    return MURDGE_MLP, depthestimator, denoiser, transfomation_param , denoise_net

def train_model(A,meas_data,config):
    main_device = config['main_device']
    depth_device = config['depth_device']
    denoiser_device = config['denoiser_device']
    lfshape = config['lfshape']
    nu,nv=lfshape[0],lfshape[1]
    #H,W=meas_data.shape[-2],meas_data.shape[-1]\
    H,W=lfshape[-2],lfshape[-1]
    lr = config['lr']

    # AMP can dramatically reduce peak activation memory for full-resolution MLP training.
    amp_enabled = bool(config.get('use_amp', True))
    amp_dtype_name = str(config.get('amp_dtype', 'bf16')).lower()
    if amp_dtype_name in ('bf16', 'bfloat16'):
        amp_dtype = torch.bfloat16
    elif amp_dtype_name in ('fp16', 'float16', 'half'):
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16

    meas_data = meas_data.to(main_device)
    
    coo_im = coo_gen((H, W)).to(main_device)
    components = return_MURDGE_componets(config)
    MURDGE_MLP, depthestimator, denoiser, transfomation_param, denoise_net=components

    start_t = time.perf_counter()
    loss_func = nn.L1Loss()
    if_rgb=(len(lfshape)==5)
    optimizer = AdamW(MURDGE_MLP.mlp_lf.parameters(), lr=lr)
#=========================S1 coarse training========================
    for iter in range(config['S1_iter']):
        # IMPORTANT: S1 still backprops through the MLP. Use chunked forward + AMP to avoid OOM.
        with torch.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
            im_raw = _forward_in_chunks(
                MURDGE_MLP.mlp_lf,
                coo_im,
                chunk_size=int(config.get('mlp_chunk_size', 2**12)),
            )

        if not if_rgb:
            im_recon = einops.rearrange(im_raw.squeeze(-1), '(h w) -> h w', h=H, w=W)
            im_recon = denoise_net(im_recon.unsqueeze(0)).squeeze()
            lf_warm = im_recon.expand(nu, nv, H, W).to(main_device)
        else:
            im_recon = einops.rearrange(im_raw, '(h w) c -> c h w ', h=H, w=W, c=3)
            im_recon = denoise_net(im_recon.unsqueeze(0)).squeeze(0)
            lf_warm = im_recon.expand(nu, nv, 3, H, W).to(main_device)
        with torch.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
            meas_hat = A(lf_warm.to(main_device, non_blocking=True))
        loss = loss_func(meas_hat.float(), meas_data.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (np.mod(iter, 20) == 0):
            print('Iter: {0:.1f} loss: {1:.5f} '.format(iter, loss.item()))



#==========================S2 setup=========================
    lf_input = lf_warm.detach().clone()
    init_disparity = depthestimator(norm(im_recon))
    disp_max_value = torch.max(torch.abs(init_disparity)).item()
    
    DEQ_MURDGEmodel = DEQ_MURDGE(MURDGE_MLP, depthestimator, denoiser, transfomation_param, lfshape, disp_max_value)
    deq = get_deq(core='sliced', ift=True)
    
    # Manually ensure bias_img and other non-MLP params are on main_device
    DEQ_MURDGEmodel.bias_img.data = DEQ_MURDGEmodel.bias_img.data.to(main_device)
    if hasattr(DEQ_MURDGEmodel, 'transfomation_param'):
         # transfomation_param is already on main_device from initialization, but good to be safe if it's a parameter
         pass
        


    DEQ_cycle = config.get('DEQ_cycle', 1)
    
    dir_path = os.path.join(config['save_dir'],'rawdata')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    if config.get('bias_start_iter', 0)!=0:
        param_groups = [
            {'params': [p for n, p in DEQ_MURDGEmodel.named_parameters() if 'bias_img' not in n], 'lr': lr},
            {'params': [p for n, p in DEQ_MURDGEmodel.named_parameters() if 'bias_img' in n], 'lr': 0.0}
        ]
        optimizer = AdamW(param_groups)
        delayed_bias_training = True
    else:
        # Standard: optimize all parameters together
        optimizer = AdamW(DEQ_MURDGEmodel.parameters(), lr=lr)
        delayed_bias_training = False
    

#==========================S2 DEQ training=========================
    for iter in range(config['S2_iter']):
        if delayed_bias_training and iter == config['bias_start_iter']:
            print(f"\n[Iter {iter}] Activating bias_img training (setting lr from 0 to {lr})")
            for param_group in optimizer.param_groups:
                if any('bias_img' in n for n, p in DEQ_MURDGEmodel.named_parameters() 
                       if any(p is param for param in param_group['params'])):
                    param_group['lr'] = lr



        #if (iter % DEQ_cycle == 0)  and (iter >= config['DEQstart_iter'] and (iter<config['S2_iter']-50)):
        if (iter % DEQ_cycle == 0)  and (iter >= config['DEQstart_iter']):
            torch.cuda.empty_cache()
            z_out, _ = deq(DEQ_MURDGEmodel, lf_input,
                            solver_kwargs=dict(f_solver='anderson'),
                            stop_mode='abs')
            lf_star = z_out[-1]
                
        else:
            with torch.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
                lf_star = DEQ_MURDGEmodel(lf_input)

        

        with torch.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
            meas_hat = A(lf_star.to(main_device, non_blocking=True))

        loss = loss_func(meas_hat.float(), meas_data.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache() 
        lf_input = lf_star.detach().clone()

        

        print(f'Iter:{iter}/{config["S2_iter"]}, loss:{loss.item():.5f}')

    stop_t = time.perf_counter()
    print('total_train_time: {}s'.format(stop_t - start_t))
    with torch.no_grad():
        final_lf, final_disparity_full = DEQ_MURDGEmodel.forward_full(lf_star)
    k = MURDGE_MLP.compute_scale_alpha()
    print(k)

    return final_lf.detach(), final_disparity_full.detach(), DEQ_MURDGEmodel 





if __name__ == '__main__':
    gpu_list = [0,1]

    num = 5
    u=v=5
    spatial_dec = 4
    save_dir = 'test'

    meas_data,A,AT = load_exp_data(num,meas_dir='ExpData_total/CodedAperture/pokemon',if_rgb=True,spatial_dec=spatial_dec)
    h,w=meas_data.shape[-2],meas_data.shape[-1] #h,w=341,426

    lfshape=[u,v,3,h,w]
#============ MURDGE Reconstruct ==============
    print(f'Start MURDGE Reconstruct...')
    config = {}

    config['save_dir']          = save_dir
    config['lfshape']           = lfshape
    config['spaticl_dec']       = spatial_dec
    config['lr']                = 1e-3
    config['S1_iter']           = 300
    config['S2_iter']           = 150  
    config['depthmodelname']    = 'depthanythingb'  # 'depthanythingb'/'depthanythingl'/'depthanythings'/'zoedepth'/'vggt'/'Metric3D'/'DepthPro'
    config['denoiser']          = 'drunet_rgb'
    config['lambda']            = 6e-2  
    config['MLPhiddenlayers']   = 3
    config['MLPhiddenfeatures'] = 256  
    config['MLPomega0']         = 10.0  
    config['MLPsigma0']         = 10.0  
    config['MLP_mirror']        = False  
    config['need_iter_disp']    = False  
    config['bias_start_iter']   = 80   
    config['DEQ_cycle']         = 1   
    config['DEQstart_iter']     = 1000
    config['vignetting']        = False    
    config['scale_alpha_mode']  = 'finetune' #'learned'
    config['gpu_list']          = gpu_list
    config['main_device']       = f'cuda:{config["gpu_list"][0]}'
    config['depth_device']      = f'cuda:{config["gpu_list"][-1]}'
    config['denoiser_device']   = f'cuda:{config["gpu_list"][-1]}'

    #with open(os.devnull, 'w') as fnull: 
        #with contextlib.redirect_stdout(fnull): 
    lf_murdge, final_disparity_full, DEQ_MURDGEmodel = train_model(A, meas_data, config)

    
    #MURDGE data save
    #1,raw data
    disparity_path = os.path.join(save_dir,'rawdata',f'disparity.pt')
    dir_path = os.path.dirname(disparity_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    lf_path = os.path.join(save_dir,'rawdata',f'murdge_lf.pt')
    torch.save(lf_murdge,lf_path)
    torch.save(final_disparity_full,disparity_path)
    #2,disparity img
    finaldisparity = disp_allview_to_disparity(final_disparity_full,mode='edge')
    depth = disparity2depth(finaldisparity,fitting_params_path='ExpData_total/CodedAperture/fitting_param_4x.pt')



    save_img(lf_murdge[u//2,v//2], os.path.join(save_dir,f'sub_murdge.png'))
    Plot(finaldisparity,cmap='RdBu',savepath=os.path.join(save_dir,f'disparity.png'))
    
    Plot(depth,cmap='RdBu',savepath=os.path.join(save_dir,f'depth.png'))
    light_field_to_gif(norm(lf_murdge), os.path.join(save_dir,f'lfgif.gif'))
    light_field_to_video_hq(norm(lf_murdge), os.path.join(save_dir,f'lfvideo.mp4'))


