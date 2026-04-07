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

    # 图像尺寸裁剪至8的倍数
    new_h = h // 8 * 8
    new_w = w // 8 * 8
    img1_cropped = img1[:, :new_h, :new_w].float()
    img2_cropped = img2[:, :new_h, :new_w].float()

    # WAFT 训练/推理默认输入范围是 0-255
    if img1_cropped.max() <= 1.5:
        img1_cropped = img1_cropped * 255.0
    if img2_cropped.max() <= 1.5:
        img2_cropped = img2_cropped * 255.0


    WAFTmodel = _load_waft_model(device=device)

    # 推断 WAFT 模型设备
    try:
        if hasattr(WAFTmodel, 'model'):
            model_device = next(WAFTmodel.model.parameters()).device
        else:
            model_device = next(WAFTmodel.parameters()).device
    except Exception:
        model_device = device if isinstance(device, torch.device) else torch.device(str(device))

    img1_batch = img1_cropped.unsqueeze(0).to(model_device)  # [1, 3, new_h, new_w]
    img2_batch = img2_cropped.unsqueeze(0).to(model_device)

    with torch.no_grad():
        output = WAFTmodel.calc_flow(img1_batch, img2_batch)

    flow = output['flow'][-1]  # [1, 2, new_h, new_w]

    flow_up = torch.nn.functional.interpolate(
        flow,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )

    flow_output = flow_up.permute(0, 2, 3, 1).contiguous()  # [1, H, W, 2]
    flow_output[..., 0] = flow_output[..., 0]
    flow_output[..., 1] = flow_output[..., 1]

    return flow_output



def imagefile2tensor(path, if_rgb=False, normalize=True):
    img = Image.open(path)
    if if_rgb:
        img = img.convert('RGB')
    else:
        img = img.convert('L')

    arr = np.array(img)  # [H,W] 或 [H,W,3]

    if normalize:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    # [H,W] -> [1,H,W]； [H,W,3] -> [3,H,W]
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
    """
    保存光场为多帧TIFF文件
    
    参数:
        light_field_tensor: torch.Tensor
            - 灰度: [u, v, h, w]
            - RGB: [u, v, 3, h, w]
        save_path: str - 保存路径
        need_index_text: bool - 是否在图像上添加索引文字
    """
    import numpy as np
    import cv2
    import tifffile as tiff

    lf = light_field_tensor.cpu().numpy()
    
    # 检测是否为RGB模式
    if lf.ndim == 5:  # RGB: (u, v, 3, h, w)
        u, v, c, h, w = lf.shape
        is_rgb = True
        if c != 3:
            raise ValueError(f"期望通道数为3 (RGB)，但得到 {c}")
    elif lf.ndim == 4:  # 灰度: (u, v, h, w)
        u, v, h, w = lf.shape
        is_rgb = False
    else:
        raise ValueError(f"期望光场形状为 [u,v,h,w] 或 [u,v,3,h,w]，但得到 {lf.shape}")
    
    scale = h / 512

    imgs = []
    for i in range(u):
        for j in range(v):
            if is_rgb:
                # RGB模式: [3, h, w] -> [h, w, 3]
                img = lf[i, j].transpose(1, 2, 0)  # (3,h,w) -> (h,w,3)
            else:
                # 灰度模式: [h, w]
                img = lf[i, j]
            
            # 归一化到 [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # 转换为 uint8
            img = (img * 255).astype(np.uint8)

            if need_index_text:
                text = f"u={i}, v={j}"
                if is_rgb:
                    # RGB图像，使用白色文字 (255, 255, 255)
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1 * scale, (255, 255, 255), 2)
                else:
                    # 灰度图像，使用标量颜色 255
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1 * scale, 255, 2)
            
            imgs.append(img)

    stack = np.stack(imgs, axis=0)
    tiff.imwrite(save_path, stack)

    print(f"Save images to '{save_path}'")
def save_img(img, path, norm=True):
    """
    保存图像到磁盘，自动处理灰度和RGB图像
    
    参数:
        img: torch.Tensor
            - 灰度: [H,W] 或 [1,H,W] 或 [1,1,H,W]
            - RGB: [3,H,W] 或 [1,3,H,W]
        path: str - 保存路径
        norm: bool - 是否归一化到[0,1]
    """
    # 转换为numpy并移到CPU
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    
    # 统一形状到 [C,H,W]
    dim = len(img.shape)
    if dim == 2:  # [H,W] -> [1,H,W]
        img = img.unsqueeze(0)
    elif dim == 4:  # [1,C,H,W] -> [C,H,W]
        img = img.squeeze(0)
    
    # 归一化
    if norm:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # 创建目录
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    transforms.ToPILImage()(img).save(path)

def norm(img):
    img = (img-img.min())/(img.max()-img.min())
    return img
def light_field_to_gif(light_field, save_path, total_time=10, need_text=True, 
                       high_quality=True, optimize=True):
    """
    将光场保存为GIF动图
    
    参数:
        light_field: torch.Tensor
            - 灰度: [u, v, h, w]
            - RGB: [u, v, 3, h, w]
        save_path: str - 保存路径
        total_time: float - 总播放时长（秒）
        need_text: bool - 是否添加索引文字
        high_quality: bool - 高质量模式（使用pillow保存，支持更好的调色板）
        optimize: bool - 是否优化GIF大小（轻微质量损失但文件更小）
    
    注意:
        - GIF格式最多支持256色，对于高质量需求建议使用 light_field_to_video_hq()
        - high_quality=True 时使用PIL保存，支持更好的颜色量化
        - high_quality=False 时使用imageio保存（旧版行为）
    """
    light_field = light_field.detach()
    # 确保 light_field 是 PyTorch Tensor
    if not isinstance(light_field, torch.Tensor):
        light_field = torch.from_numpy(light_field)

    # 如果 light_field 在 GPU 上，移到 CPU
    if light_field.is_cuda:
        light_field = light_field.cpu()

    # 检测是否为RGB模式
    if light_field.dim() == 5:  # RGB: [u, v, 3, x, y]
        u, v, c, x, y = light_field.shape
        is_rgb = True
    else:  # 灰度: [u, v, x, y]
        u, v, x, y = light_field.shape
        is_rgb = False

    total_frames = u * v
    # 每帧显示时间（毫秒）
    frame_duration_ms = int((total_time / total_frames) * 1000)
    fps = total_frames / total_time

    frames = []

    # 遍历每个子孔径图像
    for i in range(u):
        for j in range(v):
            # 获取子孔径图像
            if is_rgb:
                img = light_field[i, j, :, :, :].numpy()  # [3, x, y]
                img = np.transpose(img, (1, 2, 0))  # -> [x, y, 3] for cv2
            else:
                img = light_field[i, j, :, :].numpy()  # [x, y]

            # 归一化并转换为 uint8
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
                # 添加文字
                text = f"u={i}, v={j}"
                if is_rgb:
                    # RGB图像，使用彩色文字（白色）
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # 灰度图像，使用标量颜色
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            # 转换为PIL Image以获得更好的质量控制
            if high_quality:
                if is_rgb:
                    pil_img = Image.fromarray(img, mode='RGB')
                else:
                    pil_img = Image.fromarray(img, mode='L')
                frames.append(pil_img)
            else:
                frames.append(img)
    
    # 保存GIF
    if high_quality:
        # 使用PIL保存，支持更好的调色板量化
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=optimize,
            # 使用自适应调色板以获得最佳质量
            # 对于RGB图像，PIL会自动使用最优的256色调色板
        )
        print(f"高质量GIF已保存到 '{save_path}'")
    else:
        # 使用imageio保存（旧版行为）
        imageio.mimsave(save_path, frames, 'GIF', fps=fps, loop=0)
        print(f"GIF已保存到 '{save_path}'")

def light_field_to_video_hq(light_field, save_path, total_time=10, need_text=True,
                           quality='high', codec=None):
    """
    将光场保存为高质量视频（推荐用于无损或接近无损保存）
    
    参数:
        light_field: torch.Tensor
            - 灰度: [u, v, h, w]
            - RGB: [u, v, 3, h, w]
        save_path: str - 保存路径（.mp4 / .avi）
        total_time: float - 总播放时长（秒）
        need_text: bool - 是否添加索引文字
        quality: str - 质量级别
            - 'lossless': 无损（使用FFV1编码，需要.avi格式）
            - 'high': 高质量（H264高码率）
            - 'medium': 中等质量
        codec: str or None - 编码器（None=自动选择）
            - 'mp4v': MP4编码（兼容性最好）
            - 'avc1': H.264编码（高质量，但可能不支持）
            - 'XVID': Xvid编码（兼容性好）
            - 'FFV1': 无损编码（需要.avi，文件较大）
    
    注意:
        - 对于真正的无损，使用 quality='lossless' 和 .avi 格式
        - 对于高质量有损，使用 quality='high' 和 .mp4 格式
        - RGB和灰度均支持
        - 如果指定编码器失败，会自动回退到 mp4v（兼容性最好）
    """
    light_field = light_field.detach()
    if not isinstance(light_field, torch.Tensor):
        light_field = torch.from_numpy(light_field)
    
    if light_field.is_cuda:
        light_field = light_field.cpu()
    
    # 检测是否为RGB模式
    if light_field.dim() == 5:  # RGB: [u, v, 3, h, w]
        u, v, c, h, w = light_field.shape
        is_rgb = True
    else:  # 灰度: [u, v, h, w]
        u, v, h, w = light_field.shape
        is_rgb = False
    
    total_frames = u * v
    fps = total_frames / total_time
    
    # 自动选择编码器或设置用户指定的编码器
    if quality == 'lossless':
        codec_to_try = ['FFV1', 'mp4v']  # 无损优先，回退到mp4v
        if not save_path.endswith('.avi'):
            print("警告: 无损模式建议使用.avi格式，已自动修改扩展名")
            save_path = save_path.rsplit('.', 1)[0] + '.avi'
    elif codec is None:
        # 自动选择：尝试多个编码器，按兼容性排序
        codec_to_try = ['mp4v', 'XVID', 'MJPG', 'avc1']
    else:
        # 用户指定编码器，尝试后回退到mp4v
        codec_to_try = [codec, 'mp4v']
    
    # 尝试创建VideoWriter
    frame_size = (w, h)  # (width, height)
    out = None
    used_codec = None
    
    for codec_name in codec_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(save_path, fourcc, fps, frame_size, isColor=True)
            
            if out.isOpened():
                used_codec = codec_name
                #print(f"成功使用编码器: {codec_name}")
                break
            else:
                out.release()
                out = None
        except Exception as e:
            print(f"编码器 {codec_name} 不可用: {e}")
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError(
            f"无法创建视频文件: {save_path}\n"
            f"尝试的编码器: {codec_to_try}\n"
            f"建议: 1) 安装 ffmpeg: sudo apt install ffmpeg\n"
            f"     2) 重新编译OpenCV with ffmpeg支持\n"
            f"     3) 使用 light_field_to_gif() 作为替代"
        )
    
    frame_count = 0  # 计数器，用于调试
    
    # 写入每一帧
    for i in range(u):
        for j in range(v):
            if is_rgb:
                img = light_field[i, j, :, :, :].numpy()  # [3, h, w]
                img = np.transpose(img, (1, 2, 0))  # -> [h, w, 3]
            else:
                img = light_field[i, j, :, :].numpy()  # [h, w]
            
            # 归一化到 [0, 255]
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)
            
            img_uint8 = img.astype(np.uint8)
            
            # 确保图像是连续的内存布局
            if not img_uint8.flags['C_CONTIGUOUS']:
                img_uint8 = np.ascontiguousarray(img_uint8)
            
            if need_text:
                text = f"u={i}, v={j}"
                if is_rgb:
                    # RGB: 转BGR格式，白色文字
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    out.write(img_bgr)
                else:
                    # 灰度: 转为3通道再添加文字
                    img_gray_3ch = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    cv2.putText(img_gray_3ch, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    out.write(img_gray_3ch)
            else:
                if is_rgb:
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                else:
                    # 灰度图转为3通道
                    img_gray_3ch = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    out.write(img_gray_3ch)
            
            frame_count += 1
    
    out.release()
    print(f"高质量视频已保存到 '{save_path}'")
    #print(f"  - 质量={quality}, 编码={used_codec}, fps={fps:.2f}")
    #print(f"  - 总帧数: {frame_count}/{total_frames} (应为 {u}x{v}={total_frames})")

def light_field_to_video(light_field, total_time, save_path):
    """
    将尺寸为 (u, v, x, y) 的光场张量转换为视频。

    参数：
    - light_field: torch.FloatTensor，尺寸为 (u, v, x, y)
    - total_time: 视频总时长（秒）
    - save_path: 视频保存路径
    """
    import torch
    import numpy as np
    import cv2

    # 确保 light_field 是 torch.FloatTensor
    #assert isinstance(light_field, torch.FloatTensor), "light_field 必须是 torch.FloatTensor 类型"
    light_field = light_field.cpu().detach()
    # 获取维度
    u, v, x, y = light_field.shape
    total_frames = u * v
    frame_rate = total_frames / total_time  # 每秒帧数

    # 准备 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 对于 MP4 文件
    frame_size = (y, x)  # 宽度，高度
    # 设置 isColor=False 表示灰度图像
    out = cv2.VideoWriter(save_path, fourcc, frame_rate, frame_size, isColor=False)

    # 遍历每个子孔径图像
    for i in range(u):
        for j in range(v):
            # 获取子孔径图像
            img = light_field[i, j, :, :].numpy()

            # 归一化并转换为 uint8
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)

            img = img.astype(np.uint8)

            # 添加文字
            text = f"u={i}, v={j}"
            # 在灰度图像中，颜色参数为标量
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            # 写入帧
            out.write(img)

    # 释放视频写入器
    out.release()
def psnr_torch(img_rec, img_gt, max_val):
    if not isinstance(img_gt, torch.Tensor):
        img_gt = torch.as_tensor(img_gt, dtype=img_rec.dtype, device=img_rec.device)
    img_gt = img_gt.to(img_rec.device)
    mse = torch.nn.MSELoss()(img_rec,img_gt)
    eps = 1e-10  # 防止 log(0)
    psnr = 10 * torch.log10((max_val ** 2) / (mse + eps))
    return psnr.item()  # 输出 shape=(N,)


def save_depthimg(img, path, norm=True, cmap='RdBu'):
    """
    保存 Tensor 图像到磁盘。
    ----------
    img : torch.Tensor  [C,H,W] / [H,W] / [1,C,H,W]
    path: str          输出路径
    norm: bool         是否 0-1 归一化
    cmap: str | None   例如 'jet'、'gray'、'RdBu'…
                       None 表示不着色，保持原灰度/RGB
    """
    # ---- 1. 将形状统一成 [C,H,W] ----
    if img.dim() == 2:            # [H,W] → [1,H,W]
        img = img.unsqueeze(0)
    elif img.dim() == 4:          # [1,C,H,W] → [C,H,W]
        img = img.squeeze(0)

    # ---- 2. 拷到 CPU，并做归一化 ----
    img = img.detach().cpu().float()
    if norm:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    # ---- 3. 是否需要着色 ----
    if cmap is not None:
        # 只处理单通道；多通道直接按 RGB 保存
        assert img.shape[0] == 1, "cmap 只适用于单通道图像"
        arr = img.squeeze(0).numpy()          # [H,W] 0-1
        rgb = mpl.colormaps[cmap](arr)[..., :3]
       # rgb = cm.get_cmap(cmap)(arr)[..., :3] # [H,W,3] 去掉 alpha
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


    # 图像尺寸裁剪至8的倍数
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

    # 推断 RAFT 模型所在设备
    try:
        model_device = next(RAFTmodel.parameters()).device
    except Exception:
        # 回退到函数参数提供的 device
        model_device = device if isinstance(device, torch.device) else torch.device(str(device))

    # 添加batch维度并移动到与模型相同的设备
    img1_batch = img1_normalized.unsqueeze(0).to(model_device)  # [1, 3, new_h, new_w]
    img2_batch = img2_normalized.unsqueeze(0).to(model_device)

    # 推理
    with torch.no_grad():
        flows = RAFTmodel(img1_batch, img2_batch)

    # 获取最终flow [1, 2, new_h, new_w]
    flow = flows[-1]

    # 上采样到原始尺寸
    flow_up = torch.nn.functional.interpolate(
        flow,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )

    # 调整形状为 [1, H, W, 2]
    flow_output = flow_up.permute(0, 2, 3, 1)  # [1, H, W, 2]
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
        """
        私有方法，用于预处理图像。
        现在接收一个动态的 target_size。
        """
        if image_tensor.dim() == 2:           # (H,W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif image_tensor.dim() == 3:         # (C,H,W) or (1,H,W)
            image_tensor = image_tensor.unsqueeze(0)  # (1,C,H,W)
        if image_tensor.dim() != 4:
            raise ValueError("输入张量维度应为 2/3/4，但收到 shape: {}".format(image_tensor.shape))
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

        # 使用传入的动态 target_size进行缩放
        resized_img = F.interpolate(
            padded_img,
            size=(target_size, target_size),
            mode='bicubic',
            align_corners=False
        )

        return resized_img, (original_h, original_w), padding
    def _postprocess_depth(self, depth_map: torch.Tensor, original_hw: tuple, padding: tuple) -> torch.Tensor:
        """
        私有方法，用于后处理深度图，将其恢复到原始尺寸。
        depth_map: [H, W] 已经 squeeze 过的深度图
        输出: [H, W] 单通道深度图
        """
        # (pad_left, pad_right, pad_top, pad_bottom)
        pad_left, pad_right, pad_top, pad_bottom = padding
        original_h, original_w = original_hw

        # depth_map 应该是 [H, W]，转换为 [1, 1, H, W] 用于插值
        if depth_map.dim() != 2:
            raise ValueError(f"Expected depth_map to be [H, W], but got shape: {depth_map.shape}")
        
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]

        # 1. 将深度图从 (target_size, target_size) 放大回填充后的尺寸 (max_dim, max_dim)
        max_dim = max(original_h, original_w)
        # 对于深度图，使用 'bilinear' 插值通常效果更好，避免产生不存在的深度值
        resized_depth = F.interpolate(
            depth_map,
            size=(max_dim, max_dim),
            mode='bilinear',
            align_corners=False
        )

        # 2. 裁剪掉填充区域，恢复到原始图像尺寸
        # 计算裁剪的索引
        crop_start_h = pad_top
        crop_end_h = max_dim - pad_bottom
        crop_start_w = pad_left
        crop_end_w = max_dim - pad_right

        # 执行裁剪
        final_depth = resized_depth[:, :, crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        # 输出 [H, W]：移除 batch 和 channel 维度
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

        # 1. 加载 Marigold Pipeline

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

        # --- 1. 可微分的预处理 (复刻源码逻辑) ---
        # a) 归一化: 将输入从 [0, 1] 变换到 [-1, 1]，这是 VAE 的要求
        rgb_norm = image_tensor * 2.0 - 1.0


        if original_h%8!=0 or original_w%8!=0:
            #processing_res = min(original_h//8*8,original_w//8*8)
            h=original_h//8*8
            w=original_w//8*8


            processed_tensor = F.interpolate(
                rgb_norm,
                size=(h, w),
                #size=(processing_res, processing_res),
                mode='bicubic',
                align_corners=False
            )
        else:
            processed_tensor = rgb_norm
        processed_tensor = processed_tensor.to(self.dtype)

        # 设置随机种子
        generator = torch.Generator(device=self.device).manual_seed(seed)

        ensemble_predictions = []
        iterable = range(ensemble_size)

        with torch.no_grad():  # 推理核心仍然在 no_grad 上下文中
            for _ in iterable:
                # single_infer 返回的是一个在 [0, 1] 范围内的深度图张量
                depth_pred = self.pipe.single_infer(
                    rgb_in=processed_tensor,
                    num_inference_steps=denoising_steps,
                    generator=generator,
                    show_pbar=False
                )
                ensemble_predictions.append(depth_pred)

        # --- 3. 集成与后处理 ---
        # a) 将预测列表堆叠成一个张量
        predictions_stack = torch.stack(ensemble_predictions)  # (ensemble_size, N, 1, H_proc, W_proc)
        # 如果有批次维度，需要调整
        if n_batch > 1:
            # (ensemble_size, N, 1, H, W) -> (N, ensemble_size, 1, H, W)
            predictions_stack = predictions_stack.permute(1, 0, 2, 3, 4)
            ensembled_results = []
            for single_item_stack in predictions_stack:  # 遍历批次中的每个项目
                ensembled_pred, _ = self.ensemble_depth(single_item_stack)
                ensembled_results.append(ensembled_pred)
            depth_pred_ensembled = torch.cat(ensembled_results, dim=0)
        else:
            # 官方的集成函数
            depth_pred_ensembled, _ = self.ensemble_depth(predictions_stack.squeeze(1))  # (ensemble_size, 1, H_proc, W_proc)

        # b) 可微分的缩放: 将深度图恢复到原始尺寸
        final_depth = F.interpolate(
            depth_pred_ensembled,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )

        # 移除通道维度，输出 (N, H, W)
        return final_depth.squeeze()

class ZoeDepthEstimator:
    """
    - 输入: (H,W) / (C,H,W) / (B,C,H,W)，C=1或3，任意值域（不做归一化/缩放）
    - 输出: (B,H,W)，空间尺寸与输入一致；全可微，可对输入求梯度
    - 仅执行：尺寸对齐到32倍数 -> 前向 -> 回插值到原尺寸
    """
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
        # HF 模型参数名为 pixel_values
        outputs = self.model(pixel_values=x_in)
        depth = outputs.predicted_depth          # (B,H',W')
        depth = self._postprocess(depth, size_orig, out_dtype=self.dtype)
        return depth
class Metric3DEstimator:
    """
    - 输入: (H,W) / (C,H,W) / (B,C,H,W)，C=1或3；不做归一化，仅尺寸对齐
    - 输出: (B,H,W)，与输入空间尺寸一致
    - 逻辑: 对齐到32倍数 -> 前向 -> 回插值到原尺寸
    - 依赖: 官方 PyTorch Hub 模型 (YvanYin/Metric3D)
    """
    def __init__(self,
                 variant: str = "metric3d_vit_small",  # 也可 'metric3d_vit_large' / 'metric3d_vit_giant2' / convnext
                 device= "cuda",
                 dtype: torch.dtype = torch.float32,
                 freeze_model: bool = True):
        """
        variant 取值参考官方README: metric3d_convnext_tiny / metric3d_convnext_large /
                                   metric3d_vit_small / metric3d_vit_large / metric3d_vit_giant2
        """
        repo = "/media/users/LMH/.cache/torch/hub/YvanYin_Metric3D_main"
        self.device = device
        self.dtype = dtype

        # 通过 PyTorch Hub 加载 Metric3D
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                self.net = torch.hub.load(repo, variant, source='local',pretrain=True,trust_repo=True, force_reload=False)  # 需联网 & 首次会下权重

        self.net.to(self.device)
        self.net.eval()
        if freeze_model:
            for p in self.net.parameters():
                p.requires_grad = False

    # ---- 形状准备：接收 2D/3D/4D 输入，输出 (B,3,H,W) ----
    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        只做形状整理：
          (H,W)        -> (1,1,H,W) -> (1,3,H,W)（灰度补三通道）
          (C,H,W)      -> (1,C,H,W)（C=1或3；C=1时补三通道）
          (B,C,H,W)    -> 原样（若 C=1 则补三通道；C=3 直接用）
        不改尺寸，不归一化。
        """
        if img.dim() == 2:  # (H,W)
            img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif img.dim() == 3:  # (C,H,W)
            img = img.unsqueeze(0)  # (1,C,H,W)
        elif img.dim() != 4:
            raise ValueError(f"输入维度应为 2/3/4，收到: {tuple(img.shape)}")

        # 此时必为 (B,C,H,W)
        B, C, H, W = img.shape
        if C == 1:
            img = img.repeat(1, 3, 1, 1)  # 灰度 -> RGB
        elif C != 3:
            raise ValueError(f"期望通道数 1 或 3，收到 C={C}")

        return img.to(self.device)

    def _preprocess(self, x: torch.Tensor):
        """
        不再 resize。仅记录原始尺寸，并搬到 device/dtype。
        """
        B, C, H, W = x.shape
        size_orig = (H, W)
        x = x.to(self.device, self.dtype, non_blocking=True)
        return x, size_orig

    def _postprocess(self, depth: torch.Tensor, size_orig, out_dtype: torch.dtype):
        """
        接受 (B,1,h,w)/(B,h,w)/(h,w)/(B,C,h,w)；插值回原始 (H,W)，返回 (B,H,W)。
        """


        B,_, hp, wp = depth.shape
        H, W = size_orig

        if (hp, wp) != (H, W):
            depth = torch.nn.functional.interpolate(
                depth, size=(H, W),
                mode="bicubic", align_corners=False
            )

        return depth.to(dtype=out_dtype).squeeze()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        image 可设 requires_grad=True（但见下“可微注意”）
        返回: (B,H,W) 深度（与输入空间尺寸一致）
        """
        # 1) 整形 & 类型 & 设备
        image = self._prepare_image(image).to(self.device, self.dtype)

        # 2) 记录原尺寸并对齐到 32 的倍数
        x_in, size_orig = self._preprocess(image)   # (B,3,H32,W32)

        # 3) 前向（官方Hub推理接口）
        # 保存当前的 CUDA 设备，在推理时临时切换到模型设备，之后恢复
        current_device = None
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
        
        try:
            # 设置当前设备为模型所在设备，确保内部创建的张量在正确的设备上
            dev = torch.device(self.device)
            if dev.type == 'cuda':
                torch.cuda.set_device(dev)
            
            # 官方 README 示例：pred_depth, conf, out_dict = self.net.inference({'input': rgb})
            # 我们仅取深度图；保持梯度开着（若内部未强制 no_grad）
            with torch.set_grad_enabled(True):
                pred_depth, _, _ = self.net.inference({'input': x_in.to(self.device)})
        finally:
            # 恢复之前的 CUDA 设备
            if current_device is not None:
                torch.cuda.set_device(current_device)

        # 4) 回插值到原图大小
        depth = self._postprocess(pred_depth, size_orig, out_dtype=self.dtype)
        return depth





def Plot_psnr_curve(scene_data,path):
    DATA = scene_data
    num = DATA['num']
    #lf
    murdge_lf_psnr=DATA['murdge_lf_psnr']
    pnp_lf_psnr=DATA['pnp_lf_psnr']
    #x axis
    compression = []
    for j in range(len(num)):
        num_i = float(num[j])
        compression_i = 81 / num_i
        compression.append(compression_i)

    #===================================================plot
    # 绘图
    plt.figure(figsize=(5, 4))

    # MURDGE lf
    plt.plot(compression, murdge_lf_psnr, marker='o', color='blue', linestyle='-', label='MURDGE_lf')

    # PnP lf
    plt.plot(compression, pnp_lf_psnr, marker='s', color='orange', linestyle='-', label='PnP_lf')


    # 坐标轴设置
    plt.xlabel('Compression Factor')
    plt.ylabel('PSNR')
    plt.xscale('log')
    plt.xticks(compression, [f"{c:.0f}" for c in compression], rotation=45)

    # 刻度线优化
    # ax = plt.gca()
    # ax.xaxis.set_ticks_position('bottom')
    # ax.tick_params(which='minor', length=0)

    # 图例设置
    plt.legend(loc='best')

    # 布局优化
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图表
    #plt.show()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)  # 保存为 PNG 格式
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
    """
    读取文件夹中的所有图片，按文件名排序后，组装成光场 tensor。

    参数
    ----
    folder : str
        图片文件夹路径
    u, v : Optional[int]
        光场角分辨率。
        - 若同时提供 u 和 v，则要求图片总数 == u*v
        - 若都不提供，则自动推断为平方光场（例如 25 张 -> 5x5）
    extensions : tuple
        允许读取的图片后缀
    to_float : bool
        是否转为 float32
        - True: 输出范围 [0, 1]
        - False: 保留原始 uint8 数值
    rgb : bool
        是否强制转为 RGB
        - True: 输出 c=3
        - False: 保留原图通道，但要求所有图片通道一致

    返回
    ----
    lf : torch.Tensor
        形状 [u, v, c, h, w]

    说明
    ----
    图片排列方式默认是按排序后的顺序，先填满 v，再换到下一行 u：
    index 0 -> [0, 0]
    index 1 -> [0, 1]
    ...
    index v -> [1, 0]
    """
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

    # 收集图片文件
    img_files: List[Path] = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if len(img_files) == 0:
        raise ValueError(f"No image files found in folder: {folder}")

    # 自然排序
    img_files = sorted(img_files, key=lambda x: _natural_key(x.name))

    n = len(img_files)

    # 自动推断 u,v
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
    scan_mode="raster",                 # "raster" or "spiral"
    caption_font_size=36,
    caption_font_path=None,             # 例如 r"C:\Windows\Fonts\arialbd.ttf"
    gap_px=24,
    top_margin_px=None,
    spiral_center=None,                 # 默认自动取 (u//2, v//2)
    codec="libx264",
    crf=12,
    preset="slow",
    pix_fmt="yuv420p"
):
    """
    将多个 light field 水平拼接为一个视频。
    每个 light field 形状必须一致，均为 [u, v, c, h, w] 的 torch.Tensor。

    参数
    ----
    light_field_list : list[torch.Tensor]
        每个元素形状为 [u, v, c, h, w]
    output_path : str
        输出视频路径，例如 "results/out.mp4"
    caption_list : list[str] or None
        每个 light field 对应一个标题；None 表示不显示标题
    total_duration_sec : float
        总时长（秒），函数会自动根据总帧数调整 fps
    scan_mode : str
        "raster" 或 "spiral"
    caption_font_size : int
        标题字号
    caption_font_path : str or None
        Arial Bold 字体路径。若为 None，则尝试系统默认 Arial 粗体路径
    gap_px : int
        不同 light field 之间的水平间隔（黑色）
    top_margin_px : int or None
        标题区域高度；None 时自动估计
    spiral_center : tuple or None
        spiral 模式的中心坐标，默认自动取 (u//2, v//2)
    codec, crf, preset, pix_fmt :
        ffmpeg 编码参数
    """
    import math
    from pathlib import Path

    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    # ----------------------------
    # 1. 基本检查
    # ----------------------------
    if not isinstance(light_field_list, (list, tuple)) or len(light_field_list) == 0:
        raise ValueError("light_field_list 必须是非空 list/tuple，且内部元素为 [u,v,c,h,w] 的 tensor。")

    if total_duration_sec <= 0:
        raise ValueError("total_duration_sec 必须大于 0。")

    n_lf = len(light_field_list)

    if caption_list is not None and len(caption_list) != n_lf:
        raise ValueError("caption_list 的长度必须与 light_field_list 相同。")

    # ----------------------------
    # 2. 逐个光场做“整体归一化”
    #    不是逐帧归一化，避免破坏不同子孔径之间的亮度关系
    # ----------------------------
    normalized_lf_list = []
    for idx, lf in enumerate(light_field_list):
        lf = torch.as_tensor(lf).detach().float().cpu()

        if lf.ndim != 5:
            raise ValueError(f"第 {idx} 个 light field 的维度不是 5，当前 shape = {tuple(lf.shape)}，应为 [u,v,c,h,w]。")

        lf = torch.nan_to_num(lf, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = torch.amin(lf)
        max_val = torch.amax(lf)

        if float(max_val - min_val) < 1e-12:
            lf = torch.zeros_like(lf)
        else:
            lf = (lf - min_val) / (max_val - min_val)

        normalized_lf_list.append(lf)

    # ----------------------------
    # 3. 检查所有光场 shape 一致
    # ----------------------------
    ref_shape = tuple(normalized_lf_list[0].shape)
    for idx, lf in enumerate(normalized_lf_list):
        if tuple(lf.shape) != ref_shape:
            raise ValueError(
                f"所有 light field 的 shape 必须一致。"
                f"第 0 个为 {ref_shape}，第 {idx} 个为 {tuple(lf.shape)}"
            )

    u, v, c, h, w = ref_shape
    if c not in [1, 3]:
        raise ValueError(f"当前只支持 c=1 或 c=3，但检测到 c={c}。")

    # ----------------------------
    # 4. 生成扫描路径
    # ----------------------------
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

        if not (0 <= center_r < u and 0 <= center_c < v):
            raise ValueError(f"spiral_center={spiral_center} 超出范围，当前光场大小为 ({u}, {v})。")

        # outward spiral：右 -> 上 -> 左 -> 下
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

        # 螺旋向外后，再沿原路径反向回中心
        if len(outward) == 1:
            scan_sequence = outward
        else:
            inward = outward[-2::-1]   # 反向回中心，不重复最外层点
            scan_sequence = outward + inward

    else:
        raise ValueError("scan_mode 只能是 'raster' 或 'spiral'。")

    num_frames = len(scan_sequence)
    fps = num_frames / float(total_duration_sec)

    # ----------------------------
    # 5. 载入字体（优先 Arial Bold）
    # ----------------------------
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

        # 自动估计标题区域高度
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

    # ----------------------------
    # 6. 计算输出帧尺寸
    # ----------------------------
    frame_h = top_margin_px + h
    frame_w = n_lf * w + (n_lf - 1) * gap_px

    # H.264 + yuv420p 通常要求宽高为偶数
    final_frame_h = frame_h if frame_h % 2 == 0 else frame_h + 1
    final_frame_w = frame_w if frame_w % 2 == 0 else frame_w + 1

    # ----------------------------
    # 7. 创建输出目录并写视频
    # ----------------------------
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=codec,
        macro_block_size=None,   # 禁止 imageio 自动缩放到 16 的倍数
        ffmpeg_params=[
            "-crf", str(crf),
            "-preset", str(preset),
            "-pix_fmt", str(pix_fmt),
            "-movflags", "+faststart",
        ],
    )

    try:
        for (ii, jj) in scan_sequence:
            # 黑底画布
            canvas = np.zeros((final_frame_h, final_frame_w, 3), dtype=np.uint8)

            # ----------------------------
            # 逐个光场取当前 (ii, jj) 子孔径并水平拼接
            # ----------------------------
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

            # ----------------------------
            # 添加 caption
            # ----------------------------
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


    
    
   
    