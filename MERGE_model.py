import torch
import torch.nn as nn
import numpy as np
import random 
import einops

def numpy2cuda(ndarray_in):
    return torch.from_numpy(ndarray_in).float().cuda()
def cuda2numpy(tensor_in):
    return tensor_in.cpu().numpy()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def coo_gen(im_shape):
    x = (torch.linspace(0, im_shape[-2], steps=im_shape[-2]) - im_shape[-2]//2) /im_shape[-2] *2e0
    y = (torch.linspace(0, im_shape[-1], steps=im_shape[-1]) - im_shape[-1]//2) /im_shape[-1] *2e0
    x, y = torch.meshgrid(x.cuda(), y.cuda())
    x = einops.rearrange(x, 'nx (ny nc) ->(nx ny) nc', nc=1)
    y = einops.rearrange(y, 'nx (ny nc) ->(nx ny) nc', nc=1)
    coo = torch.cat( (y, x) , dim=1)
    return coo



class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale ** 2))  # torch.cos(omega)*torch.exp(-(scale**2))

class NeRF_Wire_multigpu(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 out_nonlin,
                 omega_0=4.0,
                 sigma_0=4.0,
                 mp_devices=None):

        super().__init__()
        layer_list = []
        layer_list.append(RealGaborLayer(input_dim, hidden_features,
                                         is_first=True, omega0=omega_0, sigma0=sigma_0))
        for i in range(hidden_layers):
            layer_list.append(RealGaborLayer(hidden_features, hidden_features,
                                             is_first=False, omega0=omega_0, sigma0=sigma_0))
        if (out_nonlin == 'sigmoid'):
            layer_list.append(nn.Linear(hidden_features, out_features))
            layer_list.append(nn.Sigmoid())
        elif (out_nonlin == 'gabor'):
            layer_list.append(RealGaborLayer(hidden_features, out_features,
                                            is_first=False, omega0=omega_0, sigma0=sigma_0))
        elif (out_nonlin == 'linear'):
            layer_list.append(nn.Linear(hidden_features, out_features))

        if mp_devices is None or len(mp_devices) == 0:
            self.segments = nn.ModuleList([nn.Sequential(*layer_list)])
            self.segment_devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
        else:
            n_dev = len(mp_devices)
            n_layers = len(layer_list)
            
            base_size = n_layers // n_dev
            remainder = n_layers % n_dev
            
            sizes = [base_size] * n_dev
            
            if remainder > 0:
                for i in range(remainder):
                    idx = n_dev - 1 - i  
                    sizes[idx] += 1
            segments = []
            idx = 0
            for seg_i, sz in enumerate(sizes):
                seg_layers = layer_list[idx: idx + sz]
                idx += sz
                dev = torch.device(mp_devices[seg_i])
                seq = nn.Sequential(*seg_layers)
                
                for module in seq.modules():
                    module.to(dev)
                
                segments.append(seq)
            
            self.segments = nn.ModuleList(segments)
            self.segment_devices = [torch.device(d) for d in mp_devices]

    def forward(self, coords):
        x = coords.to(self.segment_devices[0])
        
        for i, (seg, dev) in enumerate(zip(self.segments, self.segment_devices)):
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            seg.to(dev)
            x = seg(x)
        return x

class COLF_Wire_split_multigpu(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 N_view,
                 hash_length,
                 omega_0=5.0,
                 sigma_0=5.0,
                 mirror=False,
                 device='cuda',
                 needfinetune=True,
                 need_split=False,
                 chunk_size=32768,
                 mp_devices=None):
        super().__init__()
        
        self.H, self.W = hash_length[0], hash_length[1]
        self.N_view = N_view
        self.need_split = bool(need_split)
        self.chunk_size = int(chunk_size)
        
        if mp_devices and len(mp_devices) > 0:
            self.primary_device = torch.device(mp_devices[0])
        else:
            self.primary_device = torch.device(device)
            print(f"[COLF v2] Using single device: {self.primary_device}")
        
        if needfinetune:
            self.beta = nn.Parameter(torch.zeros((N_view, 2), device=self.primary_device))
        else:
            self.beta = torch.zeros((N_view, 2), device=self.primary_device)
        
        N_u = np.int32(np.sqrt(N_view))
        N_v = N_u
        self.delta = N_u / (N_u - 1) / 10
        
        if not mirror:
            v_c = (np.linspace(N_v, 0.0, N_v) - N_v / 2.0)
            u_c = (np.linspace(0.0, N_u, N_u) - N_u / 2.0)
        else:
            u_c = (np.linspace(N_v, 0.0, N_v) - N_v / 2.0)
            v_c = (np.linspace(0.0, N_u, N_u) - N_u / 2.0)
        
        xu, yv = np.meshgrid(u_c, v_c)
        xu = xu.flatten()
        yv = yv.flatten()
        self.scale_alpha = numpy2cuda(np.vstack((xu, yv)).T).to(self.primary_device)

        self.k = nn.Parameter(torch.ones(1, device=self.primary_device))
        
        if mp_devices is not None and len(mp_devices) > 0:
            self.mlp_lf = NeRF_Wire_multigpu(
                input_dim=input_dim,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=out_features,
                out_nonlin='sigmoid',
                omega_0=omega_0,
                sigma_0=sigma_0,
                mp_devices=mp_devices
            )

    
    def compute_scale_alpha(self):
        """计算微调后的视角缩放系数"""
        device = self.scale_alpha.device
        beta_device = self.beta.to(device)
        scale_alpha2 = self.scale_alpha + torch.tanh(beta_device) * self.delta
        return scale_alpha2
    
    def forward(self, coords, disparity=None):
        """
        前向传播
        
        参数:
            coords: 图像坐标 [h*w, 2]
            disparity: 视差图 [h, w]
        
        返回:
            lf_recon_raw: 重建的光场 [N_view*h*w, out_features]
            disparity_allview: 所有视角的视差偏移 [N_view*h*w, 2]
        """
        if hasattr(self.mlp_lf, 'segment_devices'):
            device = self.mlp_lf.segment_devices[0]
        else:
            device = next(self.mlp_lf.parameters()).device
        
        coords = coords.to(device)
        disparity = disparity.to(device)
        
        disp_img_flt = disparity.unsqueeze(0).unsqueeze(0)
        disparity_flt = torch.cat((disp_img_flt.squeeze(0).squeeze(0).unsqueeze(-1),
                                   self.k.to(device) * disp_img_flt.squeeze(0).squeeze(0).unsqueeze(-1)), dim=2)
        disparity_flt = einops.rearrange(disparity_flt, 'h w xy -> (h w) xy')
        
        scale_alpha2 = self.compute_scale_alpha().to(device)
        
        if not self.need_split:
            for K in range(self.N_view):
                if K == 0:
                    coord_lf = scale_alpha2[K, :].to(device) * disparity_flt + coords
                elif K == self.N_view // 2:
                    coord_lf = torch.cat((coord_lf, coords), dim=0)
                else:
                    coord_lf = torch.cat((coord_lf, scale_alpha2[K, :].to(device) * disparity_flt + coords), dim=0)
            coord_lf=torch.clip(coord_lf,-1.,1.)
            lf = self.mlp_lf(coord_lf)
            return lf, coord_lf - coords.repeat(self.N_view, 1)
        
        outputs = []
        N_pixels = disparity_flt.shape[0]
        offset_chunks_all = []
        
        for K in range(self.N_view):
            if K == self.N_view // 2:
                term = coords
                offset_this_view = torch.zeros_like(disparity_flt, device=device)
            else:
                alpha = scale_alpha2[K, :].to(device)
                offset_this_view = alpha * disparity_flt
                term = offset_this_view + coords.to(device)
            
            out_chunks = []
            for i in range(0, N_pixels, self.chunk_size):
                chunk = term[i: i + self.chunk_size].to(device)
                chunk=torch.clip(chunk,-1.,1.)
                out_chunks.append(self.mlp_lf(chunk))
            out_view = torch.cat(out_chunks, dim=0) 
            
            outputs.append(out_view)
            offset_chunks_all.append(offset_this_view)
        
        lf_recon_raw = torch.cat(outputs, dim=0)  
        disparity_allview = torch.cat(offset_chunks_all, dim=0) 
        
        return lf_recon_raw, disparity_allview

class COLF_Wire_rand_multigpu(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 N_view,
                 hash_length,
                 omega_0=5.0,
                 sigma_0=5.0,
                 mirror=False,
                 device='cuda',
                 needfinetune=True,
                 need_split=False,
                 chunk_size=32768,
                 mp_devices=None):
        super().__init__()
        
        self.H, self.W = hash_length[0], hash_length[1]
        self.N_view = N_view
        self.need_split = bool(need_split)
        self.chunk_size = int(chunk_size)
        
        if mp_devices and len(mp_devices) > 0:
            self.primary_device = torch.device(mp_devices[0])
        else:
            self.primary_device = torch.device(device)
            print(f"[COLF v2] Using single device: {self.primary_device}")
        

        self.k = torch.ones(1, device=self.primary_device)
        self.scale_alpha = nn.Parameter(1e-4 *torch.randn((N_view, 2), device=self.primary_device))
        if mp_devices is not None and len(mp_devices) > 0:
            self.mlp_lf = NeRF_Wire_multigpu(
                input_dim=input_dim,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=out_features,
                out_nonlin='sigmoid',
                omega_0=omega_0,
                sigma_0=sigma_0,
                mp_devices=mp_devices
            )
    
    def compute_scale_alpha(self):
        scale_alpha2 = self.scale_alpha
        return scale_alpha2
    
    def forward(self, coords, disparity=None):
        if hasattr(self.mlp_lf, 'segment_devices'):
            device = self.mlp_lf.segment_devices[0]
        else:
            device = next(self.mlp_lf.parameters()).device
        
        coords = coords.to(device)
        disparity = disparity.to(device)
        
        disp_img_flt = disparity.unsqueeze(0).unsqueeze(0)
        disparity_flt = torch.cat((disp_img_flt.squeeze(0).squeeze(0).unsqueeze(-1),
                                   self.k.to(device) * disp_img_flt.squeeze(0).squeeze(0).unsqueeze(-1)), dim=2)
        disparity_flt = einops.rearrange(disparity_flt, 'h w xy -> (h w) xy')
        
        scale_alpha2 = self.compute_scale_alpha().to(device)
        
        if not self.need_split:
            for K in range(self.N_view):
                if K == 0:
                    coord_lf = scale_alpha2[K, :].to(device) * disparity_flt + coords
                elif K == self.N_view // 2:
                    coord_lf = torch.cat((coord_lf, coords), dim=0)
                else:
                    coord_lf = torch.cat((coord_lf, scale_alpha2[K, :].to(device) * disparity_flt + coords), dim=0)
            
            lf = self.mlp_lf(coord_lf)
            return lf, coord_lf - coords.repeat(self.N_view, 1)
        
        outputs = []
        N_pixels = disparity_flt.shape[0]
        offset_chunks_all = []
        
        for K in range(self.N_view):
            if K == self.N_view // 2:
                term = coords
                offset_this_view = torch.zeros_like(disparity_flt, device=device)
            else:
                alpha = scale_alpha2[K, :].to(device)
                offset_this_view = alpha * disparity_flt
                term = offset_this_view + coords.to(device)
            
            out_chunks = []
            for i in range(0, N_pixels, self.chunk_size):
                chunk = term[i: i + self.chunk_size].to(device)
                out_chunks.append(self.mlp_lf(chunk))
            out_view = torch.cat(out_chunks, dim=0) 
            
            outputs.append(out_view)
            offset_chunks_all.append(offset_this_view)
        
        lf_recon_raw = torch.cat(outputs, dim=0)  
        disparity_allview = torch.cat(offset_chunks_all, dim=0)  
        
        return lf_recon_raw, disparity_allview

