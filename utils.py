import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import pandas as pd
#import open3d as o3d
import types
import torchvision
import matplotlib.cm as cm
import tifffile
from torch import nn
import math
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F
import sys
import cv2
from scipy.io import loadmat
from pathlib import Path
import einops
import kornia
import os
import imageio
from PIL import Image
from skimage.util import montage
from torchvision import transforms
from torch.utils.checkpoint import checkpoint
#from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import lpips,contextlib
#from transformers import AutoModelForDepthEstimation
from LFBM5D_GPU import LFBM5D_denoiser as lfbm5d
import json
import importlib
from typing import Optional, Tuple, List
# mpl.use('TkAgg')
device='cuda'
#RAFT model
#weights = Raft_Large_Weights.DEFAULT
#RAFTmodel = raft_large(weights=weights, progress=False).to(device)
#RAFTmodel.eval()
RAFTmodel=None



WAFT_MODEL = None
WAFT_MODEL_META = None

def lf_downsample(lf: torch.Tensor, downsample_factor, mode='bilinear', align_corners=False):


    u, v, c, h, w = lf.shape
    # bilinear/bicubic interpolation in PyTorch requires floating-point tensors
    lf_flat = einops.rearrange(lf, 'u v c h w -> (u v) c h w').float()
    lf_down_flat = F.interpolate(
            lf_flat,
            scale_factor=1/downsample_factor,
            mode=mode,
            align_corners=align_corners,
            antialias=True
    )

    lf_downsampled = einops.rearrange(lf_down_flat, '(u v) c h w -> u v c h w',u=u, v=v)
    return lf_downsampled
def _load_waft_model(
    cfg_path='Github_proj/WAFT-waftv2/config/a2/twins/chairs-things.json',
    ckpt_path='Github_proj/WAFT-waftv2/ckpts/twins/zero-shot.pth',
    device=device,
    scale=0.0,
    pad_to_train_size=False,
    tiling=False,
):
    global WAFT_MODEL, WAFT_MODEL_META

    target_device = device if isinstance(device, torch.device) else torch.device(str(device))
    cfg_path_obj = Path(cfg_path)
    ckpt_path_obj = Path(ckpt_path)
    if not cfg_path_obj.is_absolute():
        cfg_path_obj = Path.cwd() / cfg_path_obj
    if not ckpt_path_obj.is_absolute():
        ckpt_path_obj = Path.cwd() / ckpt_path_obj
    cfg_path_obj = cfg_path_obj.resolve()
    ckpt_path_obj = ckpt_path_obj.resolve()

    meta = (
        str(cfg_path_obj),
        str(ckpt_path_obj),
        str(target_device),
        float(scale),
        bool(pad_to_train_size),
        bool(tiling),
    )
    if WAFT_MODEL is not None and WAFT_MODEL_META == meta:
        return WAFT_MODEL

    if not cfg_path_obj.exists():
        raise FileNotFoundError(f'WAFT config not found: {cfg_path_obj}')
    if not ckpt_path_obj.exists():
        raise FileNotFoundError(f'WAFT checkpoint not found: {ckpt_path_obj}')

    waft_root = cfg_path_obj.parent
    while waft_root != waft_root.parent:
        if (waft_root / 'model').is_dir() and (waft_root / 'inference_tools.py').exists():
            break
        waft_root = waft_root.parent
    else:
        raise FileNotFoundError(
            f'Cannot locate WAFT project root from config path: {cfg_path_obj}'
        )

    if str(waft_root) not in sys.path:
        sys.path.insert(0, str(waft_root))

    with open(cfg_path_obj, 'r', encoding='utf-8') as f:
        args_dict = json.load(f)
    args = SimpleNamespace(**args_dict)

    existing_utils = sys.modules.get('utils')
    existing_utils_utils = sys.modules.get('utils.utils')
    existing_model = sys.modules.get('model')
    existing_config = sys.modules.get('config')
    existing_inference_tools = sys.modules.get('inference_tools')

    try:
        waft_utils_pkg = types.ModuleType('utils')
        waft_utils_pkg.__path__ = [str(waft_root / 'utils')]
        sys.modules['utils'] = waft_utils_pkg
        if 'utils.utils' in sys.modules:
            del sys.modules['utils.utils']
        if 'model' in sys.modules:
            del sys.modules['model']
        if 'config' in sys.modules:
            del sys.modules['config']
        if 'inference_tools' in sys.modules:
            del sys.modules['inference_tools']

        waft_model_module = importlib.import_module('model')
        waft_utils_module = importlib.import_module('utils.utils')
        waft_inference_module = importlib.import_module('inference_tools')

        model = waft_model_module.fetch_model(args)
        waft_utils_module.load_ckpt(model, str(ckpt_path_obj))
        model = model.to(target_device)
        model.eval()

        train_size = getattr(args, 'image_size', None)
        wrapper = waft_inference_module.InferenceWrapper(
            model,
            scale=float(scale),
            train_size=train_size,
            pad_to_train_size=bool(pad_to_train_size),
            tiling=bool(tiling),
        )
    finally:
        if existing_utils is not None:
            sys.modules['utils'] = existing_utils
        elif 'utils' in sys.modules:
            del sys.modules['utils']

        if existing_utils_utils is not None:
            sys.modules['utils.utils'] = existing_utils_utils
        elif 'utils.utils' in sys.modules:
            del sys.modules['utils.utils']

        if existing_model is not None:
            sys.modules['model'] = existing_model
        elif 'model' in sys.modules and not hasattr(sys.modules['model'], '__path__'):
            del sys.modules['model']

        if existing_config is not None:
            sys.modules['config'] = existing_config
        elif 'config' in sys.modules and not hasattr(sys.modules['config'], '__path__'):
            del sys.modules['config']

        if existing_inference_tools is not None:
            sys.modules['inference_tools'] = existing_inference_tools

    WAFT_MODEL = wrapper
    WAFT_MODEL_META = meta
    return WAFT_MODEL

def compute_disparity_with_waft(img1_tensor, img2_tensor, device=device):  #[j,w]
    img1_tensor = img1_tensor.squeeze()
    img2_tensor = img2_tensor.squeeze()

    if img1_tensor.dim() == 3:
        c, h, w = img1_tensor.shape
        if c == 1:
            img1 = img1_tensor.repeat(3, 1, 1)
            img2 = img2_tensor.repeat(3, 1, 1)
        else:
            img1 = img1_tensor[:3]
            img2 = img2_tensor[:3]
    else:
        h, w = img1_tensor.shape
        img1 = img1_tensor.unsqueeze(0).repeat(3, 1, 1)
        img2 = img2_tensor.unsqueeze(0).repeat(3, 1, 1)

    new_h = h // 8 * 8
    new_w = w // 8 * 8
    img1_cropped = img1[:, :new_h, :new_w].float()
    img2_cropped = img2[:, :new_h, :new_w].float()

    if img1_cropped.max() <= 1.5:
        img1_cropped = img1_cropped * 255.0
    if img2_cropped.max() <= 1.5:
        img2_cropped = img2_cropped * 255.0


    WAFTmodel = _load_waft_model(device=device)

    try:
        if hasattr(WAFTmodel, 'model'):
            model_device = next(WAFTmodel.model.parameters()).device
        else:
            model_device = next(WAFTmodel.parameters()).device
    except Exception:
        model_device = device if isinstance(device, torch.device) else torch.device(str(device))

    img1_batch = img1_cropped.unsqueeze(0).to(model_device)
    img2_batch = img2_cropped.unsqueeze(0).to(model_device)

    with torch.no_grad():
        output = WAFTmodel.calc_flow(img1_batch, img2_batch)

    flow = output['flow'][-1]  

    flow_up = torch.nn.functional.interpolate(
        flow,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )

    flow_output = flow_up.permute(0, 2, 3, 1).contiguous() 
    flow_output[..., 0] = flow_output[..., 0]
    flow_output[..., 1] = flow_output[..., 1]

    return flow_output



def imagefile2tensor(path, if_rgb=False, normalize=True):
    img = Image.open(path)
    if if_rgb:
        img = img.convert('RGB')
    else:
        img = img.convert('L')

    arr = np.array(img) 

    if normalize:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)


    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    else:
        arr = arr.transpose(2, 0, 1)

    return torch.from_numpy(arr)


def Plot(filter,cmap='gray',savepath=None,is_rgb=False):
    # Check if the input is a torch.Tensor or numpy ndarray
    if isinstance(filter, torch.Tensor):
        filter = filter.detach().cpu().numpy()  # Convert to numpy if it's a tensor
    elif isinstance(filter, np.ndarray):
        pass  # It's already a numpy array, no need to change
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    # Determine if the image is RGB or grayscale
    dim = len(filter.shape)
    
    if dim == 4:  # [1, C, H, W]
        filter = filter.squeeze(0)  # -> [C, H, W]
        dim = 3
    
    if dim == 3:
        # Check if it's RGB [3, H, W] or grayscale [1, H, W]
        if filter.shape[0] == 3 and is_rgb:
            # Transpose to [H, W, 3] for imshow
            filter = np.transpose(filter, (1, 2, 0))
        elif filter.shape[0] == 3 and (is_rgb==False):
            # Transpose to [3, H, W] --> [H,W]
            filter = np.mean(filter, axis=0)
        elif filter.shape[0] == 1:
            filter = filter.squeeze(0)  # -> [H, W]
        else:
            raise ValueError(f"Unexpected channel dimension: {filter.shape[0]}. Expected 1 or 3.")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the filter on the axis
    if is_rgb:
        # For RGB images, display without cmap and without colorbar
        im = ax.imshow(filter, interpolation='nearest')
        ax.axis('off')  # Hide axis
    else:
        # For grayscale images, use cmap and add colorbar
        im = ax.imshow(filter, cmap=cmap, interpolation='nearest')
        
        # Add colorbar with legend showing only min and max values
        norm = Normalize(vmin=np.min(filter), vmax=np.max(filter))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
        cbar.set_label('Value')
        
        # Set colorbar to display only the min and max values
        cbar.set_ticks([np.min(filter), np.max(filter)])
        
        ax.axis('off')  # Hide axis
    
    if savepath==None:
        plt.show()
    else:
        dir_path = os.path.dirname(savepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(savepath,dpi=300)
        plt.close()


def save_light_field_as_tif(light_field_tensor,
                            save_path='light_field_images.tif',
                            need_index_text=False):
    import numpy as np
    import cv2
    import tifffile as tiff

    lf = light_field_tensor.cpu().numpy()
    
    if lf.ndim == 5:  
        u, v, c, h, w = lf.shape
        is_rgb = True
        if c != 3:
            raise ValueError(f"Expected channel count of 3 (RGB), but got {c}")
    elif lf.ndim == 4: 
        u, v, h, w = lf.shape
        is_rgb = False
    else:
        raise ValueError(f"Expected light field shape of [u,v,h,w] or [u,v,3,h,w], but got {lf.shape}")
    
    scale = h / 512

    imgs = []
    for i in range(u):
        for j in range(v):
            if is_rgb:
                img = lf[i, j].transpose(1, 2, 0)  # (3,h,w) -> (h,w,3)
            else:
                img = lf[i, j]
            
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            img = (img * 255).astype(np.uint8)

            if need_index_text:
                text = f"u={i}, v={j}"
                if is_rgb:

                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1 * scale, (255, 255, 255), 2)
                else:

                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1 * scale, 255, 2)
            
            imgs.append(img)

    stack = np.stack(imgs, axis=0)
    tiff.imwrite(save_path, stack)

    print(f"Save images to '{save_path}'")
def save_img(img, path, norm=True):

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    

    dim = len(img.shape)
    if dim == 2:  # [H,W] -> [1,H,W]
        img = img.unsqueeze(0)
    elif dim == 4:  # [1,C,H,W] -> [C,H,W]
        img = img.squeeze(0)
    

    if norm:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    

    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    transforms.ToPILImage()(img).save(path)

def norm(img):
    img = (img-img.min())/(img.max()-img.min())
    return img
def light_field_to_gif(light_field, save_path, total_time=10, need_text=True, 
                       high_quality=True, optimize=True):

    light_field = light_field.detach()

    if not isinstance(light_field, torch.Tensor):
        light_field = torch.from_numpy(light_field)


    if light_field.is_cuda:
        light_field = light_field.cpu()


    if light_field.dim() == 5:  # RGB: [u, v, 3, x, y]
        u, v, c, x, y = light_field.shape
        is_rgb = True
    else:  
        u, v, x, y = light_field.shape
        is_rgb = False

    total_frames = u * v

    frame_duration_ms = int((total_time / total_frames) * 1000)
    fps = total_frames / total_time

    frames = []

    for i in range(u):
        for j in range(v):

            if is_rgb:
                img = light_field[i, j, :, :, :].numpy()  # [3, x, y]
                img = np.transpose(img, (1, 2, 0))  # -> [x, y, 3] for cv2
            else:
                img = light_field[i, j, :, :].numpy()  # [x, y]


            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)
            
            img_uint8 = img.astype(np.uint8)
            del img
            img = np.ascontiguousarray(img_uint8)

            if need_text:

                text = f"u={i}, v={j}"
                if is_rgb:

                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:

                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            if high_quality:
                if is_rgb:
                    pil_img = Image.fromarray(img, mode='RGB')
                else:
                    pil_img = Image.fromarray(img, mode='L')
                frames.append(pil_img)
            else:
                frames.append(img)
    

    if high_quality:

        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=optimize,

        )
    else:

        imageio.mimsave(save_path, frames, 'GIF', fps=fps, loop=0)
    print(f"GIF have been saved to '{save_path}'")

def light_field_to_video_hq(light_field, save_path, total_time=10, need_text=True,
                           quality='high', codec=None):

    light_field = light_field.detach()
    if not isinstance(light_field, torch.Tensor):
        light_field = torch.from_numpy(light_field)
    
    if light_field.is_cuda:
        light_field = light_field.cpu()
    
    if light_field.dim() == 5: 
        u, v, c, h, w = light_field.shape
        is_rgb = True
    else:  
        u, v, h, w = light_field.shape
        is_rgb = False
    
    total_frames = u * v
    fps = total_frames / total_time
    
    if quality == 'lossless':
        codec_to_try = ['FFV1', 'mp4v'] 
        if not save_path.endswith('.avi'):
            save_path = save_path.rsplit('.', 1)[0] + '.avi'
    elif codec is None:
        codec_to_try = ['mp4v', 'XVID', 'MJPG', 'avc1']
    else:
        codec_to_try = [codec, 'mp4v']
    
    frame_size = (w, h)  
    out = None
    used_codec = None
    
    for codec_name in codec_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(save_path, fourcc, fps, frame_size, isColor=True)
            
            if out.isOpened():
                used_codec = codec_name
                break
            else:
                out.release()
                out = None
        except Exception as e:
            print(f"encoder {codec_name} unavailable: {e}")
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError(
            f"saving file failed: {save_path}"
        )
    
    frame_count = 0  
    
    for i in range(u):
        for j in range(v):
            if is_rgb:
                img = light_field[i, j, :, :, :].numpy()  # [3, h, w]
                img = np.transpose(img, (1, 2, 0))  # -> [h, w, 3]
            else:
                img = light_field[i, j, :, :].numpy()  # [h, w]
            
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)
            
            img_uint8 = img.astype(np.uint8)
            
            if not img_uint8.flags['C_CONTIGUOUS']:
                img_uint8 = np.ascontiguousarray(img_uint8)
            
            if need_text:
                text = f"u={i}, v={j}"
                if is_rgb:

                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    out.write(img_bgr)
                else:

                    img_gray_3ch = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    cv2.putText(img_gray_3ch, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    out.write(img_gray_3ch)
            else:
                if is_rgb:
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                else:

                    img_gray_3ch = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    out.write(img_gray_3ch)
            
            frame_count += 1
    
    out.release()
    print(f"video saving at '{save_path}'")


def light_field_to_video(light_field, total_time, save_path):

    import torch
    import numpy as np
    import cv2


    light_field = light_field.cpu().detach()

    u, v, x, y = light_field.shape
    total_frames = u * v
    frame_rate = total_frames / total_time  


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    frame_size = (y, x)  # 

    out = cv2.VideoWriter(save_path, fourcc, frame_rate, frame_size, isColor=False)

    for i in range(u):
        for j in range(v):
            img = light_field[i, j, :, :].numpy()

            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)

            img = img.astype(np.uint8)

            text = f"u={i}, v={j}"

            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


            out.write(img)


    out.release()
def psnr_torch(img_rec, img_gt, max_val):
    if not isinstance(img_gt, torch.Tensor):
        img_gt = torch.as_tensor(img_gt, dtype=img_rec.dtype, device=img_rec.device)
    img_gt = img_gt.to(img_rec.device)
    mse = torch.nn.MSELoss()(img_rec,img_gt)
    eps = 1e-10 
    psnr = 10 * torch.log10((max_val ** 2) / (mse + eps))
    return psnr.item() 


def save_depthimg(img, path, norm=True, cmap='RdBu'):

    if img.dim() == 2:          
        img = img.unsqueeze(0)
    elif img.dim() == 4:         
        img = img.squeeze(0)

    img = img.detach().cpu().float()
    if norm:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    if cmap is not None:
        assert img.shape[0] == 1, "cmap only supports single-channel images"
        arr = img.squeeze(0).numpy()         
        rgb = mpl.colormaps[cmap](arr)[..., :3]
        pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        pil_img = transforms.ToPILImage()(img)

    pil_img.save(path)

def compute_disparity_with_raft(img1_tensor, img2_tensor , RAFTmodel=RAFTmodel, device=device):  #[j,w]
    img1_tensor=img1_tensor.squeeze()
    img2_tensor = img2_tensor.squeeze()
    if img1_tensor.dim()==3:
        _,h,w=img1_tensor.shape
        img1 = img1_tensor
        img2 = img2_tensor
    else:
        h,w=img1_tensor.shape
        img1 = img1_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        img2 = img2_tensor.unsqueeze(0).repeat(3, 1, 1)

    new_h = h // 8 * 8
    new_w = w // 8 * 8
    img1_cropped = img1[:, :new_h, :new_w]
    img2_cropped = img2[:, :new_h, :new_w]

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img1_normalized = transform(img1_cropped)
    img2_normalized = transform(img2_cropped)

    try:
        model_device = next(RAFTmodel.parameters()).device
    except Exception:

        model_device = device if isinstance(device, torch.device) else torch.device(str(device))


    img1_batch = img1_normalized.unsqueeze(0).to(model_device)  
    img2_batch = img2_normalized.unsqueeze(0).to(model_device)


    with torch.no_grad():
        flows = RAFTmodel(img1_batch, img2_batch)


    flow = flows[-1]


    flow_up = torch.nn.functional.interpolate(
        flow,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )


    flow_output = flow_up.permute(0, 2, 3, 1) 
    flow_output[...,0] = flow_output[...,0]
    flow_output[..., 1] = flow_output[..., 1]

    return flow_output






#=============================================================depth estimator=======================================================
def get_depthanything_model(encoder,device='cuda'):
    """
    encoder = 'vits', 'vitb', 'vitl'
    """
    from DepthAnything2.depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depthanything_model = DepthAnythingV2(**model_configs[encoder])
    path = f'DepthAnything2/checkpoints/depth_anything_v2_{encoder}.pth'

    depthanything_model.load_state_dict(torch.load(path, map_location='cpu'))
    depthanything_model = depthanything_model.to(device).eval()
    return depthanything_model


class VGGTEstimator:
    def __init__(
        self,
        checkpoint_path = "DepthEstimateModel/vggt-main/checkpoints",
        vggt_root = 'DepthEstimateModel/vggt-main',
        device = "cuda",
        dtype: torch.dtype = torch.float32):
        super().__init__()
        from pathlib import Path
        self.device=device
        vggt_root = Path(vggt_root).resolve()
        sys.path.insert(0, str(vggt_root))

        from vggt.models.vggt import VGGT


        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.model = VGGT.from_pretrained("facebook/VGGT-1B",cache_dir=checkpoint_path).to(device)
        # 保证评估模式
        self.model.eval()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image=image.to(self.device)
        original_h, original_w = image.shape[-2:]
        max_dim = max(original_h, original_w)
        dynamic_target_size = int(math.ceil(max_dim / 14) * 14)
        processed_img, original_hw, padding_info = self._preprocess(image, target_size=dynamic_target_size)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.model(processed_img)



        depth_raw = predictions["depth"].squeeze()

        final_depth = self._postprocess_depth(depth_raw, original_hw, padding_info)

        return final_depth  # 应该是 [H, W]



    def _preprocess(self, image_tensor: torch.Tensor, target_size: int):
        if image_tensor.dim() == 2:           # (H,W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif image_tensor.dim() == 3:         # (C,H,W) or (1,H,W)
            image_tensor = image_tensor.unsqueeze(0)  # (1,C,H,W)
        if image_tensor.dim() != 4:
            raise ValueError("expected 2/3/4 dimensions, but received shape: {}".format(image_tensor.shape))
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)


        original_h, original_w = image_tensor.shape[-2:]

        max_dim = max(original_h, original_w)
        pad_left = (max_dim - original_w) // 2
        pad_right = max_dim - original_w - pad_left
        pad_top = (max_dim - original_h) // 2
        pad_bottom = max_dim - original_h - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded_img = F.pad(image_tensor, padding, mode='constant', value=0)

        resized_img = F.interpolate(
            padded_img,
            size=(target_size, target_size),
            mode='bicubic',
            align_corners=False
        )

        return resized_img, (original_h, original_w), padding
    def _postprocess_depth(self, depth_map: torch.Tensor, original_hw: tuple, padding: tuple) -> torch.Tensor:

        # (pad_left, pad_right, pad_top, pad_bottom)
        pad_left, pad_right, pad_top, pad_bottom = padding
        original_h, original_w = original_hw


        if depth_map.dim() != 2:
            raise ValueError(f"Expected depth_map to be [H, W], but got shape: {depth_map.shape}")
        
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]


        max_dim = max(original_h, original_w)

        resized_depth = F.interpolate(
            depth_map,
            size=(max_dim, max_dim),
            mode='bilinear',
            align_corners=False
        )


        crop_start_h = pad_top
        crop_end_h = max_dim - pad_bottom
        crop_start_w = pad_left
        crop_end_w = max_dim - pad_right

        final_depth = resized_depth[:, :, crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        final_depth = final_depth.squeeze(0).squeeze(0)

        return final_depth


class DepthAnything3Estimator:
    def __init__(
        self,
        model_name = "DA3MONO-LARGE",
        device = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        from depth_anything_3.api import DepthAnything3
        self.device = device
        
        # Load the wrapper but we will use the underlying model directly
        #DA3NESTED-GIANT-LARGE/DA3-GIANT/DA3-LARGE/DA3-BASE/DA3-SMALL/DA3METRIC-LARGE/DA3MONO-LARGE
        da3_wrapper = DepthAnything3.from_pretrained('depth-anything/'+model_name).to(device)
        self.model = da3_wrapper.model
        self.model.eval()
        
        # Determine dtype for autocast
        if torch.cuda.is_available():
             self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
             self.dtype = torch.float16

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        original_h, original_w = image.shape[-2:]
        
        # Preprocess using DepthAnything3 logic (upper_bound_resize)
        processed_img, original_hw = self._preprocess(image, target_size=504)
        
        # Prepare input for DA3: (B, N, 3, H, W)
        # processed_img is (N, 3, H, W) or (1, 3, H, W) from _preprocess
        # We treat it as a single batch of N views: (1, N, 3, H, W)
        model_input = processed_img.unsqueeze(0)
        
        # Run model
        device_type = self.device.type if isinstance(self.device, torch.device) else self.device
        with torch.autocast(device_type=device_type, dtype=self.dtype):
             predictions = self.model(
                model_input,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="saddle_balanced"
            )
        
        # Output depth shape: (B, N, 1, H, W) -> (1, N, 1, H, W)
        depth_raw = predictions["depth"]
        
        # Select center view
        N = depth_raw.shape[1]
        center_idx = N // 2
        depth_raw = depth_raw[:, center_idx, ...] # (1, 1, H, W)
        
        # Squeeze batch (dim 0) and channel (dim 1)
        depth_raw = depth_raw.squeeze(0).squeeze(0) # (H, W)
        
        final_depth = self._postprocess_depth(depth_raw, original_hw)
        
        return final_depth

    def _preprocess(self, image_tensor: torch.Tensor, target_size: int = 504):
        """
        Mimics DepthAnything3 InputProcessor 'upper_bound_resize' logic using differentiable ops.
        1. Resize longest side to target_size (default 504).
        2. Resize to nearest multiple of 14.
        3. Normalize (ImageNet).
        """
        if image_tensor.dim() == 2:           # (H,W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif image_tensor.dim() == 3:         # (C,H,W) or (1,H,W)
            image_tensor = image_tensor.unsqueeze(0)  # (1,C,H,W)
        
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        h, w = image_tensor.shape[-2:]

        # 1. Resize longest side to target_size
        scale = target_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        
        # Use bicubic for image resizing
        resized_img = F.interpolate(image_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
        
        # 2. Resize to nearest multiple of 14
        patch_size = 14
        def nearest_multiple(x, p):
            return int(round(x / p)) * p
            
        final_h = nearest_multiple(new_h, patch_size)
        final_w = nearest_multiple(new_w, patch_size)
        
        if final_h != new_h or final_w != new_w:
            resized_img = F.interpolate(resized_img, size=(final_h, final_w), mode='bicubic', align_corners=False)
            
        # 3. Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
        
        normalized_img = (resized_img - mean) / std

        return normalized_img, (h, w)

    def _postprocess_depth(self, depth_map: torch.Tensor, original_hw: tuple) -> torch.Tensor:
        """
        Resizes depth map back to original resolution.
        """
        original_h, original_w = original_hw
        
        # depth_map is (B, H, W)
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0) # (1, H, W)
            
        depth_map = depth_map.unsqueeze(1) # (B, 1, H, W)
        
        final_depth = F.interpolate(depth_map, size=(original_h, original_w), mode='bilinear', align_corners=False)
        
        return final_depth.squeeze()


class MarigoldEstimator:
    def __init__(
            self,
            model_path: str = "DepthEstimateModel/Marigold/checkpoint",
            device: str = "cuda",
            dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        from DepthEstimateModel.Marigold.marigold import MarigoldPipeline
        from DepthEstimateModel.Marigold.marigold.util.ensemble import ensemble_depth
        self.ensemble_depth = ensemble_depth


        self.pipe = MarigoldPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)



    def __call__(
            self,
            image_tensor: torch.Tensor,
            denoising_steps: int = 10,
            ensemble_size: int = 10,
            seed: int = 0
    ) -> torch.Tensor:
        if image_tensor.dim() != 4:
            raise ValueError(f"输入张量必须是4维 (N, C, H, W)，但得到 {image_tensor.dim()} 维。")

        n_batch, _, original_h, original_w = image_tensor.shape

        rgb_norm = image_tensor * 2.0 - 1.0


        if original_h%8!=0 or original_w%8!=0:

            h=original_h//8*8
            w=original_w//8*8


            processed_tensor = F.interpolate(
                rgb_norm,
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )
        else:
            processed_tensor = rgb_norm
        processed_tensor = processed_tensor.to(self.dtype)


        generator = torch.Generator(device=self.device).manual_seed(seed)

        ensemble_predictions = []
        iterable = range(ensemble_size)

        with torch.no_grad():
            for _ in iterable:

                depth_pred = self.pipe.single_infer(
                    rgb_in=processed_tensor,
                    num_inference_steps=denoising_steps,
                    generator=generator,
                    show_pbar=False
                )
                ensemble_predictions.append(depth_pred)


        predictions_stack = torch.stack(ensemble_predictions)  # (ensemble_size, N, 1, H_proc, W_proc)

        if n_batch > 1:
            # (ensemble_size, N, 1, H, W) -> (N, ensemble_size, 1, H, W)
            predictions_stack = predictions_stack.permute(1, 0, 2, 3, 4)
            ensembled_results = []
            for single_item_stack in predictions_stack:  
                ensembled_pred, _ = self.ensemble_depth(single_item_stack)
                ensembled_results.append(ensembled_pred)
            depth_pred_ensembled = torch.cat(ensembled_results, dim=0)
        else:

            depth_pred_ensembled, _ = self.ensemble_depth(predictions_stack.squeeze(1))  # (ensemble_size, 1, H_proc, W_proc)


        final_depth = F.interpolate(
            depth_pred_ensembled,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )


        return final_depth.squeeze()

class ZoeDepthEstimator:

    def __init__(self,
                 ckpt: str = "Intel/zoedepth-nyu-kitti",
                 device="cuda",
                 dtype: torch.dtype = torch.float32,
                 freeze_model: bool = True):
        self.device = device
        self.dtype = dtype
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                self.model = AutoModelForDepthEstimation.from_pretrained(ckpt)
        self.model.to(self.device).eval()
        if freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False

    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 2:             # (H,W) -> (1,1,H,W)
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:           # (C,H,W) -> (1,C,H,W)
            img = img.unsqueeze(0)
        elif img.dim() != 4:
            raise ValueError(f"输入维度应为2/3/4，收到: {tuple(img.shape)}")
        if img.shape[1] == 1:          # 灰度转三通道
            img = img.repeat(1, 3, 1, 1)
        elif img.shape[1] != 3:
            raise ValueError(f"期望通道数1或3，收到C={img.shape[1]}")
        return img

    def _preprocess(self, x: torch.Tensor):
        # 只做 32 倍数对齐；不做任何归一化/值域操作
        B, C, H, W = x.shape
        H32 = (H + 31) // 32 * 32
        W32 = (W + 31) // 32 * 32
        if (H32, W32) != (H, W):
            x = F.interpolate(x, size=(H32, W32), mode="bilinear", align_corners=False)
        return x.to(self.device, self.dtype), (H, W)

    def _postprocess(self, depth_pred: torch.Tensor, size_orig, out_dtype: torch.dtype):
        B, Hp, Wp = depth_pred.shape
        H, W = size_orig
        if (Hp, Wp) != (H, W):
            depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=(H, W),
                                       mode="bicubic", align_corners=False).squeeze(1)
        return depth_pred.to(out_dtype).squeeze()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prepare_image(image).to(self.device, self.dtype)
        x_in, size_orig = self._preprocess(image)
        outputs = self.model(pixel_values=x_in)
        depth = outputs.predicted_depth          # (B,H',W')
        depth = self._postprocess(depth, size_orig, out_dtype=self.dtype)
        return depth
class Metric3DEstimator:

    def __init__(self,
                 variant: str = "metric3d_vit_small",  
                 device= "cuda",
                 dtype: torch.dtype = torch.float32,
                 freeze_model: bool = True):

        repo = "/media/users/LMH/.cache/torch/hub/YvanYin_Metric3D_main"
        self.device = device
        self.dtype = dtype

        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                self.net = torch.hub.load(repo, variant, source='local',pretrain=True,trust_repo=True, force_reload=False)  

        self.net.to(self.device)
        self.net.eval()
        if freeze_model:
            for p in self.net.parameters():
                p.requires_grad = False

    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:

        if img.dim() == 2:  # (H,W)
            img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif img.dim() == 3:  # (C,H,W)
            img = img.unsqueeze(0)  # (1,C,H,W)
        elif img.dim() != 4:
            raise ValueError(f"输入维度应为 2/3/4，收到: {tuple(img.shape)}")

        B, C, H, W = img.shape
        if C == 1:
            img = img.repeat(1, 3, 1, 1)  
        elif C != 3:
            raise ValueError(f"期望通道数 1 或 3，收到 C={C}")

        return img.to(self.device)

    def _preprocess(self, x: torch.Tensor):

        B, C, H, W = x.shape
        size_orig = (H, W)
        x = x.to(self.device, self.dtype, non_blocking=True)
        return x, size_orig

    def _postprocess(self, depth: torch.Tensor, size_orig, out_dtype: torch.dtype):


        B,_, hp, wp = depth.shape
        H, W = size_orig

        if (hp, wp) != (H, W):
            depth = torch.nn.functional.interpolate(
                depth, size=(H, W),
                mode="bicubic", align_corners=False
            )

        return depth.to(dtype=out_dtype).squeeze()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:


        image = self._prepare_image(image).to(self.device, self.dtype)


        x_in, size_orig = self._preprocess(image)   # (B,3,H32,W32)


        current_device = None
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
        
        try:

            dev = torch.device(self.device)
            if dev.type == 'cuda':
                torch.cuda.set_device(dev)

            with torch.set_grad_enabled(True):
                pred_depth, _, _ = self.net.inference({'input': x_in.to(self.device)})
        finally:

            if current_device is not None:
                torch.cuda.set_device(current_device)

        depth = self._postprocess(pred_depth, size_orig, out_dtype=self.dtype)
        return depth





def Plot_psnr_curve(scene_data,path):
    DATA = scene_data
    num = DATA['num']

    murdge_lf_psnr=DATA['murdge_lf_psnr']
    pnp_lf_psnr=DATA['pnp_lf_psnr']

    compression = []
    for j in range(len(num)):
        num_i = float(num[j])
        compression_i = 81 / num_i
        compression.append(compression_i)


    plt.figure(figsize=(5, 4))

    # MURDGE lf
    plt.plot(compression, murdge_lf_psnr, marker='o', color='blue', linestyle='-', label='MURDGE_lf')

    # PnP lf
    plt.plot(compression, pnp_lf_psnr, marker='s', color='orange', linestyle='-', label='PnP_lf')


    plt.xlabel('Compression Factor')
    plt.ylabel('PSNR')
    plt.xscale('log')
    plt.xticks(compression, [f"{c:.0f}" for c in compression], rotation=45)

    plt.legend(loc='best')

    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    #plt.show()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300) 
    plt.close()

def disp_allview_to_disparity(disp_allview,mode='edge'):
    """
        return pixel disparity map [H,W]
    """

    n,h,w,c = disp_allview.shape #c=2
    u=v=int(np.sqrt(n))
    disp_allview=einops.rearrange(disp_allview,'(u v) h w c -> u v h w c',u=u,v=v)
    if mode=='stereo':
        disp_A = disp_allview[u//2,v//2,:,: ,0]  #[2,2]
        disp_B = disp_allview[u//2,-1,:,: ,0]   #[2,4]
        disp=disp_B-disp_A
    elif mode=='edge':
        disp_A = disp_allview[0,0,:,: ,0]  #[2,2]
        disp_B = disp_allview[-1,-1,:,: ,0]   #[2,4]
        disp=disp_B-disp_A
    finaldisp=disp*w/2
    return finaldisp.squeeze()
def disparity2depth(disparity,fitting_params_path='ExpData_total/CodedAperture/fitting_param_4x.pt'):

    params = torch.load(fitting_params_path, weights_only=False)
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']

    depth =  beta / (disparity + alpha) + gamma
    return depth


def rmse_torch(recon_depth, gt_depth):
    if recon_depth.device != gt_depth.device:
        recon_depth = recon_depth.to(gt_depth.device)
    
    rmse = torch.sqrt(torch.mean((recon_depth - gt_depth) ** 2))
    return rmse.item()
def dir2lf(
    folder: str,
    u = None,
    v = None,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'),
    to_float: bool = True,
    rgb: bool = True
) -> torch.Tensor:
    import re
    from typing import List
    def _natural_key(s: str):
        return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    img_files: List[Path] = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if len(img_files) == 0:
        raise ValueError(f"No image files found in folder: {folder}")

    img_files = sorted(img_files, key=lambda x: _natural_key(x.name))

    n = len(img_files)

    if u is None and v is None:
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(
                f"Number of images = {n}, cannot automatically infer square light field.\n"
                f"Please specify u and v manually."
            )
        u, v = side, side
    elif u is None or v is None:
        raise ValueError("Please provide both u and v, or neither of them.")

    if u * v != n:
        raise ValueError(
            f"Number of images ({n}) does not match u*v ({u}*{v}={u*v})."
        )

    imgs = []
    ref_h, ref_w, ref_c = None, None, None

    for img_path in img_files:
        img = Image.open(img_path)

        if rgb:
            img = img.convert('RGB')
            arr = np.array(img)  # [h, w, 3]
        else:
            arr = np.array(img)
            if arr.ndim == 2:
                arr = arr[:, :, None]  # [h, w, 1]
            elif arr.ndim != 3:
                raise ValueError(f"Unsupported image shape in {img_path}: {arr.shape}")

        h, w, c = arr.shape

        if ref_h is None:
            ref_h, ref_w, ref_c = h, w, c
        else:
            if (h, w, c) != (ref_h, ref_w, ref_c):
                raise ValueError(
                    f"Image size/channel mismatch:\n"
                    f"Reference: {(ref_h, ref_w, ref_c)}\n"
                    f"Current:   {(h, w, c)} in file {img_path.name}"
                )

        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [c, h, w]

        if to_float:
            tensor = tensor.float() / 255.0

        imgs.append(tensor)

    # [n, c, h, w] -> [u, v, c, h, w]
    lf = torch.stack(imgs, dim=0).view(u, v, ref_c, ref_h, ref_w)

    return lf
def light_field_list_to_video(
    light_field_list,
    output_path,
    caption_list=None,
    total_duration_sec=6.0,
    scan_mode="raster",             
    caption_font_size=36,
    caption_font_path=None,           
    gap_px=24,
    top_margin_px=None,
    spiral_center=None,                 
    codec="libx264",
    crf=12,
    preset="slow",
    pix_fmt="yuv420p"
):

    import math
    from pathlib import Path

    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    if not isinstance(light_field_list, (list, tuple)) or len(light_field_list) == 0:
        raise ValueError("light_field_list must not be an empty list/tuple, shape should be [u,v,c,h,w]")

    if total_duration_sec <= 0:
        raise ValueError("total_duration_sec should > 0。")

    n_lf = len(light_field_list)

    if caption_list is not None and len(caption_list) != n_lf:
        raise ValueError(" length of caption_list should match light_field_list")

    normalized_lf_list = []
    for idx, lf in enumerate(light_field_list):
        lf = torch.as_tensor(lf).detach().float().cpu()

        if lf.ndim != 5:
            raise ValueError(f"The order {idx} light field's dim is not 5, current shape = {tuple(lf.shape)}, should be [u,v,c,h,w]。")

        lf = torch.nan_to_num(lf, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = torch.amin(lf)
        max_val = torch.amax(lf)

        if float(max_val - min_val) < 1e-12:
            lf = torch.zeros_like(lf)
        else:
            lf = (lf - min_val) / (max_val - min_val)

        normalized_lf_list.append(lf)

    ref_shape = tuple(normalized_lf_list[0].shape)
    for idx, lf in enumerate(normalized_lf_list):
        if tuple(lf.shape) != ref_shape:
            raise ValueError(
                f"The order zero's shape is {ref_shape}, the order {idx}'s shape is {tuple(lf.shape)}"
            )

    u, v, c, h, w = ref_shape
    if c not in [1, 3]:
        raise ValueError(f"Only support c=1 or c=3,but got c={c}。")

    scan_mode = scan_mode.lower()
    scan_sequence = []

    if scan_mode in ["raster", "flat", "row", "row-major"]:
        for i in range(u):
            for j in range(v):
                scan_sequence.append((i, j))

    elif scan_mode == "spiral":
        if spiral_center is None:
            center_r, center_c = u // 2, v // 2
        else:
            center_r, center_c = spiral_center


        visited = set()
        outward = []

        r, cc = center_r, center_c
        if (0 <= r < u) and (0 <= cc < v):
            outward.append((r, cc))
            visited.add((r, cc))

        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # right, up, left, down
        step_len = 1
        dir_idx = 0

        while len(outward) < u * v:
            for _ in range(2):
                dr, dc = directions[dir_idx % 4]
                for _ in range(step_len):
                    r += dr
                    cc += dc
                    if 0 <= r < u and 0 <= cc < v and (r, cc) not in visited:
                        outward.append((r, cc))
                        visited.add((r, cc))
                        if len(outward) == u * v:
                            break
                dir_idx += 1
                if len(outward) == u * v:
                    break
            step_len += 1


        if len(outward) == 1:
            scan_sequence = outward
        else:
            inward = outward[-2::-1]  
            scan_sequence = outward + inward

    else:
        raise ValueError("scan_mode 只能是 'raster' 或 'spiral'。")

    num_frames = len(scan_sequence)
    fps = num_frames / float(total_duration_sec)

    use_caption = caption_list is not None
    font = None

    if use_caption:
        font_candidates = []
        if caption_font_path is not None:
            font_candidates.append(str(caption_font_path))

        font_candidates += [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "Arialbd.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/mnt/c/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

        for fp in font_candidates:
            try:
                font = ImageFont.truetype(fp, caption_font_size)
                break
            except Exception:
                pass

        if font is None:
            font = ImageFont.load_default()

        if top_margin_px is None:
            dummy_img = Image.new("RGB", (100, 100), (0, 0, 0))
            dummy_draw = ImageDraw.Draw(dummy_img)
            max_text_h = 0
            for text in caption_list:
                text = str(text)
                try:
                    bbox = dummy_draw.textbbox((0, 0), text, font=font)
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_h = caption_font_size
                max_text_h = max(max_text_h, text_h)
            top_margin_px = max(16, int(math.ceil(max_text_h * 1.8)))
    else:
        top_margin_px = 0

    frame_h = top_margin_px + h
    frame_w = n_lf * w + (n_lf - 1) * gap_px

    final_frame_h = frame_h if frame_h % 2 == 0 else frame_h + 1
    final_frame_w = frame_w if frame_w % 2 == 0 else frame_w + 1

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=codec,
        macro_block_size=None,   
        ffmpeg_params=[
            "-crf", str(crf),
            "-preset", str(preset),
            "-pix_fmt", str(pix_fmt),
            "-movflags", "+faststart",
        ],
    )

    try:
        for (ii, jj) in scan_sequence:
            canvas = np.zeros((final_frame_h, final_frame_w, 3), dtype=np.uint8)

            for lf_idx, lf in enumerate(normalized_lf_list):
                img = lf[ii, jj]  # [c,h,w]
                img = img.clamp(0, 1)

                if c == 1:
                    img = img.repeat(3, 1, 1)

                img = img.permute(1, 2, 0).contiguous().numpy()   # [h,w,3]
                img = (img * 255.0 + 0.5).astype(np.uint8)

                x0 = lf_idx * (w + gap_px)
                y0 = top_margin_px
                canvas[y0:y0 + h, x0:x0 + w] = img



            if use_caption:
                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)

                for lf_idx, text in enumerate(caption_list):
                    text = str(text)
                    x0 = lf_idx * (w + gap_px)

                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        text_x = int(round(x0 + w / 2.0 - text_w / 2.0 - bbox[0]))
                        text_y = int(round((top_margin_px - text_h) / 2.0 - bbox[1]))
                    except Exception:
                        text_w = len(text) * caption_font_size * 0.6
                        text_h = caption_font_size
                        text_x = int(round(x0 + w / 2.0 - text_w / 2.0))
                        text_y = int(round((top_margin_px - text_h) / 2.0))

                    draw.text(
                        (text_x, text_y),
                        text,
                        font=font,
                        fill=(255, 255, 255)
                    )

                canvas = np.asarray(pil_img)

            writer.append_data(canvas)

    finally:
        writer.close()


    
    