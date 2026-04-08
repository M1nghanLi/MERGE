import os
import pandas as pd
import torch
import numpy as np
import time
from torch.optim import Adam, AdamW

from MERGE_model import *
from CreateData import *
from utils import *

from torchdeq.core import get_deq
#from Fast_REDPRO_solver import *
from DenoisingLIB import *

gpu_list          = [0]
main_device = torch.device(f'cuda:{gpu_list[0]}' if torch.cuda.is_available() else 'cpu')
depth_device = torch.device(f'cuda:{gpu_list[0]}' if torch.cuda.is_available() else 'cpu')
denoiser_device = torch.device(f'cuda:{gpu_list[-1]}' if torch.cuda.is_available() else 'cpu')

class DEQ_MERGE(nn.Module):
    def __init__(self, MERGE_MLP, depthestimator, denoiser, transfomation_param, lfshape,disp_max_value):
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
            

        
        disparity_relative = self.depthestimator(guide_img).squeeze(0) /self.disp_max_value
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
                guide_img = lf[u//2,v//2].clone().unsqueeze(0) 
                bias_on_device = self.bias_img.to(guide_img.device, non_blocking=True)
                guide_img = guide_img + bias_on_device
            
            disparity_relative = self.depthestimator(guide_img).squeeze(0) /self.disp_max_value
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
        
        total_layers = 1 + hiddenlayers + 2  
        
    except Exception as e:
        print(f"[GPU Strategy] Error: {e}")
        mp_devices = []


    if len(mp_devices) > 0 :
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
    elif depthmodelname=='zoedepth':
        depthmodel = ZoeDepthEstimator(device=depth_device)
        depthestimator = lambda x: 1 / depthmodel(x)
    elif depthmodelname=='vggt':
        depthmodel = VGGTEstimator(device=depth_device)
        depthestimator = lambda x: 1 / depthmodel(x)
    else:
        raise ValueError('Unsupported depth model name')
    


    denoise_net = load_denoiser(config)

    if not if_rgb:
        denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv h w -> (nu nv) h w')), '(nu nv) h w -> nu nv h w', nu=u)
    else:
        denoiser = lambda lf: einops.rearrange(denoise_net(einops.rearrange(lf, 'nu nv c h w -> (nu nv) c h w')), '(nu nv) c h w -> nu nv c h w', nu=u,c=3)
    

    
    #transfomation_param
    transfomation_param = torch.ones(2, dtype=torch.float32, device=main_device) * 1e-2
    transfomation_param.requires_grad_(True)

    return MERGE_MLP, depthestimator, denoiser, transfomation_param , denoise_net

def train_model(A,meas_data,config):
    main_device = config['main_device']
    lfshape = config['lfshape']
    nu,nv=lfshape[0],lfshape[1]
    H,W=meas_data.shape[-2],meas_data.shape[-1]
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
    init_disparity = depthestimator(im_recon)
    disp_max_value = torch.max(torch.abs(init_disparity)).item()
    DEQ_MERGEmodel = DEQ_MERGE(MERGE_MLP, depthestimator, denoiser, transfomation_param, lfshape, disp_max_value).to(main_device)
    deq = get_deq(core='sliced', ift=True)
    

    optimizer = AdamW(DEQ_MERGEmodel.parameters(), lr=lr)

    DEQ_cycle = config.get('DEQ_cycle', 1)


    if config.get('bias_start_iter', 0)!=0:
        param_groups = [
            {'params': [p for n, p in DEQ_MERGEmodel.named_parameters() if 'bias_img' not in n], 'lr': lr},
            {'params': [p for n, p in DEQ_MERGEmodel.named_parameters() if 'bias_img' in n], 'lr': 0.0}
        ]
        optimizer = AdamW(param_groups)
        delayed_bias_training = True
    else:
        optimizer = AdamW(DEQ_MERGEmodel.parameters(), lr=lr)
        delayed_bias_training = False


#==========================S2 DEQ training=========================
    for iter in range(config['S2_iter']):
        if delayed_bias_training and iter == config['bias_start_iter']:
            print(f"\n[Iter {iter}] Activating bias_img training (setting lr from 0 to {lr})")
            for param_group in optimizer.param_groups:
                if any('bias_img' in n for n, p in DEQ_MERGEmodel.named_parameters() 
                       if any(p is param for param in param_group['params'])):
                    param_group['lr'] = lr



        if (iter % DEQ_cycle == 0)  and (iter >= config['DEQstart_iter'] and (iter<config['S2_iter']-50)):
            torch.cuda.empty_cache()
            z_out, _ = deq(DEQ_MERGEmodel, lf_input,
                            solver_kwargs=dict(f_solver='anderson'),
                            stop_mode='abs')
            lf_star = z_out[-1]
            with torch.no_grad():
                residual = torch.mean(torch.abs((lf_star-DEQ_MERGEmodel(lf_star))))
                lf_input = lf_star.detach().clone()
                
        else:
            lf_star = DEQ_MERGEmodel(lf_input)
            with torch.no_grad():
                residual = torch.mean(torch.abs((lf_star-lf_input)))
        
        meas_hat = A(lf_star.to(main_device, non_blocking=True))
        loss = loss_func(meas_hat, meas_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lf_input = lf_star.detach().clone()

        

        print(f'Iter:{iter}/{config["S2_iter"]}, loss:{loss.item():.5f} residual:{residual.item():.2e}')

    stop_t = time.perf_counter()
    print('total_train_time: {}s'.format(stop_t - start_t))
    with torch.no_grad():
        final_lf, final_disparity_full = DEQ_MERGEmodel.forward_full(lf_star)
    k = MERGE_MLP.compute_scale_alpha()
    print(k)

    return final_lf.detach(), final_disparity_full.detach(), DEQ_MERGEmodel 


def PowerMethod_for_MaxLambda(Iteration, A, At, lightFieldResolution,device='cuda'):
    LF_now = torch.rand(lightFieldResolution).to(device)
    lambda_now = torch.tensor(0.,device=device)
    with torch.no_grad():
        for i in range(Iteration):

            LF_next = A(LF_now)

            LF_next = At(LF_next)

            lambda_next = torch.sum(LF_next * LF_now) / (torch.sum(LF_now ** 2)+1e-10)

            LF_next = LF_next/torch.norm(LF_next,p='fro')

            print(f'Epoch [{i + 1}/{Iteration},lambda={lambda_next.item():.6f},|lambda_k+1-lambda_k|={torch.abs(lambda_next-lambda_now).item():.6f}]')
            LF_now=LF_next.clone()
            lambda_now = lambda_next.clone()
    return lambda_now

def PnPreconstruct(lfproj_observed, lf_init, A, At ,config, denoiser):
    mu = config['mu']
    max_iters = config['N_iters']
    device = config['device']
    lfproj_observed = lfproj_observed.to(device, non_blocking=True)
    if isinstance(mu, torch.Tensor):
        mu = float(mu.detach().cpu().item())
    else:
        mu = float(mu)
    num = lfproj_observed.shape[0]
    alpha=config['alpha']
    denoise_batch = config['denoise_batch']
    is_rgb = (lf_init.dim() == 5)
    u,v = lf_init.shape[:2]
    h,w = lf_init.shape[-2:]

    with torch.no_grad():

        qt_pre=1.0
        xt = lf_init.to(device)
        xt_pre = lf_init.to(device)
        st_pre = lf_init.to(device)

        for i in range(max_iters):
            iter_start_time = time.time()

            lfproj_hat = A(st_pre)
            grad = At(lfproj_hat-lfproj_observed)

            loss = 0.5 * torch.sum((lfproj_hat - lfproj_observed) ** 2)/ num


            st_pre = st_pre - mu * grad
            st_pre.clamp_(min=0)


            maxv = st_pre.max()
            minv = st_pre.min()
            st_pre = (st_pre-minv)/(maxv-minv)

            st_pre_denoised = denoiser(st_pre)

            xt = alpha * st_pre_denoised + (1 - alpha) * st_pre
            xt = xt*(maxv-minv)+minv



            qt = 0.5 * (1 + math.sqrt(1.0 + 4 * (qt_pre ** 2)))
            st_pre = xt + ((qt_pre - 1) / qt) * (xt - xt_pre)
            qt_pre = qt
            xt_pre = xt

            print(f'Epoch [{i + 1}/{max_iters}], Loss: {loss.item():.3e}, Time: {time.time() - iter_start_time:.2f}s')
    return xt


if __name__ == '__main__':
    num_list=[4]
    scene_list=['cotton.pt']

    dir = 'ExpData_total/LFdataset/9x9'
    save_root = 'output_test'

    for times in range(1):
        save_dir = os.path.join(save_root,f'times{times}')
        print(f'++++++++++++++++++++++++++++++++++++++times={times}++++++++++++++++++++++++++++++++++++++')
        for i in range(len(scene_list)):
            scene_name = os.path.splitext(scene_list[i])[0]
            torch.cuda.empty_cache()
            print(f'    ==========================={scene_name}===========================  ')
            path = os.path.join(dir, scene_list[i])

            for j in range(len(num_list)):
                print(f'      -----------num={num_list[j]}-----------      ')
                num = num_list[j]
                meas_data, A, AT, lfshape, lf_data, meas_data_pnp, lf_data_pnp = get_CSLIPMeasurement(path,num=num,spatial_dec=3)

                u,v,_,h,w=lfshape
    #============ MERGE Reconstruct ==============
                #print(f'Start MERGE Reconstruct...')
                config = {}
                config['main_device']       = main_device
                config['depth_device']      = depth_device
                config['denoiser_device']   = denoiser_device
                config['save_dir']          = save_dir
                config['lfshape']           = lfshape
                config['lr']                = 1e-3
                config['S1_iter']           = 600
                config['S2_iter']           = 300   

                config['depthmodelname']    = 'depthanythingb'  # 'depthanythingb'/'depthanythingl'/'depthanythings'/'zoedepth'/'vggt'
                config['denoiser']          = 'ffdnet_rgb'
                config['lambda']            = 6e-2  
                config['MLPhiddenlayers']   = 3
                config['MLPhiddenfeatures'] = 128   
                config['MLPomega0']         = 10.0  
                config['MLPsigma0']         = 10.0  
                config['bias_start_iter']   = 80
                config['DEQ_cycle']         = 10
                config['DEQstart_iter']     = 1500  # no explicit DEQ training
                config['gpu_list']          = gpu_list  

                with open(os.devnull, 'w') as fnull: 
                    with contextlib.redirect_stdout(fnull): 
                        lf_MERGE, final_disparity_full, DEQ_MERGEmodel = train_model(A,meas_data.to(main_device),config)
                
                #MERGE data save
                #1,raw data
                disparity_path = os.path.join(save_dir,'rawdata',f'disparity_{scene_name}_num{num}.pt')
                dir_path = os.path.dirname(disparity_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                lf_path = os.path.join(save_dir,'rawdata',f'{scene_name}_num{num}_MERGE_lf.pt')
                torch.save(lf_MERGE,lf_path)
                finaldisparity = disp_allview_to_disparity(final_disparity_full)

                torch.save(finaldisparity,disparity_path)
                torch.save(final_disparity_full,os.path.join(save_dir,'rawdata',f'fulldisparity_{scene_name}_num{num}.pt'))
                
                psnr_MERGE = psnr_torch(lf_MERGE, lf_data, 0.5)


    #============ PnP Reconstruct ==============
                #print(f'Start PnP Reconstruct...')
                with open(os.devnull, 'w') as fnull:  
                    with contextlib.redirect_stdout(fnull): 
                        lambda_max = PowerMethod_for_MaxLambda(200, A, AT, lfshape)
                gamma = 1 / lambda_max * 0.99
                config2 = {}
                config2['mu'] = gamma
                config2['N_iters'] = 300
                if num>30:
                    config2['alpha'] = 1e-3
                elif num > 15:
                    config2['alpha'] = 1.5e-3
                else:
                    config2['alpha'] = 2e-3
                config2['denoise_batch'] = 25
                config2['device'] = main_device
                lf_init = torch.rand(lfshape)
                # config_pnp = {}
                # config_pnp['denoiser']  = 'ffdnet_rgb'
                # config_pnp['lambda']    = 30/255.0
                # config_pnp['device']    = denoiser_device
                # Mydenoiser = load_denoiser(config_pnp)

                Mydenoiser = lambda x: lfbm5d(x,1e-1)
                with open(os.devnull, 'w') as fnull:  
                    with contextlib.redirect_stdout(fnull):  
                        lf_recon_pnp = PnPreconstruct(meas_data_pnp,lf_init,A, AT, config2, Mydenoiser)
                
                pnplf_path = os.path.join(save_dir,'rawdata',f'{scene_name}_num{num}_pnp_lf.pt')
                dir_path = os.path.dirname(pnplf_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(lf_recon_pnp,pnplf_path)

                psnr_pnp = psnr_torch(lf_recon_pnp, lf_data_pnp, 1)


    #============== if or not save picture ==============
                lf_MERGE = norm(lf_MERGE.detach())
                lf_recon_pnp = norm(lf_recon_pnp.detach())

                save_img(lf_MERGE[u//2,v//2], os.path.join(save_dir,'picture',f'{scene_name}',f'sub_num{num}_MERGE.png'))
                save_img(lf_recon_pnp[u//2,v//2], os.path.join(save_dir,'picture',f'{scene_name}',f'sub_num{num}_pnp.png'))

                save_depthimg(finaldisparity,os.path.join(save_dir,'picture',f'{scene_name}',f'disparity_num{num}.png'))

                light_field_to_gif(lf_MERGE, os.path.join(save_dir,'picture',f'{scene_name}',f'lf_MERGE_num{num}.gif'))
                light_field_to_gif(lf_recon_pnp, os.path.join(save_dir,'picture',f'{scene_name}',f'lf_pnp_num{num}.gif'))

                print(f'PSNR of MERGE: {psnr_MERGE:.2f} dB, PSNR of PnP: {psnr_pnp:.2f} dB')



