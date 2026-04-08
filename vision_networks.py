from __future__ import print_function
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
import warnings
warnings.filterwarnings("ignore")
import os
import torch.nn as nn

def load_model(device=None):
    model_path = os.getcwd() + '/model_zoo/ffdnet_gray.pth'
    #model_path = os.getcwd() + '/model_zoo/ircnn_gray.pth'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # ----------------------------------------
    # load model
    n_channels = 1        # setting for grayscale image
    nc = 64               # setting for grayscale image
    nb = 15               # setting for grayscale image
    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model
def load_model_ffdnetrgb(device=None):
    model_path = os.getcwd() + '/model_zoo/ffdnet_color.pth'
    #model_path = os.getcwd() + '/model_zoo/ircnn_gray.pth'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # ----------------------------------------
    # load model
    n_channels = 3        # setting for RGB image
    nc = 96               # setting for RGB image
    nb = 12               # setting for RGB image
    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def load_model_DRUNet():
    model_path = os.getcwd() + '/model_zoo/drunet_gray.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    n_channels = 1        # setting for grayscale image
    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, 
                act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def load_model_DRUNet_rgb(device=None):
    model_path = os.getcwd() + '/model_zoo/drunet_color.pth'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # ----------------------------------------
    # load model
    n_channels = 3        # setting for color image
    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, 
                act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def load_model_Restormer():
    model_path = os.getcwd() + '/model_zoo/restormer_gaussian_gray_denoising_sigma50.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    from models.restormer_arch import Restormer
    inp_channels=1 
    out_channels=1 
    dim = 48
    num_blocks = [4,6,6,8]
    num_refinement_blocks = 4
    heads = [1,2,4,8]
    ffn_expansion_factor = 2.66
    bias = False
    LayerNorm_type = 'BiasFree'   ## Other option 'BiasFree'
    dual_pixel_task = False   
    
    model = Restormer(inp_channels=inp_channels, out_channels= out_channels, dim=dim, num_blocks=num_blocks,
                      num_refinement_blocks=num_refinement_blocks, heads = heads, ffn_expansion_factor=ffn_expansion_factor,
                      bias = bias, LayerNorm_type=LayerNorm_type,dual_pixel_task=dual_pixel_task )
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def load_model_KBNet():
    ''' Denoising gray images with KBNet
        Note that the image size should be multiples of 8
        Also, KBNet is very sensitive to the matching of noise levels
    '''
    model_path = os.getcwd() + '/model_zoo/kbnet_gau_gray_50.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    from models.kbnet_s_arch import KBNet_s
    
    model = KBNet_s(img_channel=1, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=False, ffn_scale=2)
    
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['net'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    
    return model

def load_model_swinIR():
    args = {}
    args['task'] ='gray_dn'      # help=('classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car, color_jpeg_car')
    args['scale'] = 2                   # help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    args['noise'] = 15                  # help='noise level: 15, 25, 50')
    args['jpeg'] = 40                   # help='scale factor: 10, 20, 30, 40')
    args['model_path'] = os.getcwd() + '/model_zoo/grayDN_DFWB_s128w8_SwinIR-M_noise25.pth'# 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth' #  004_grayDN_DFWB_s128w8_SwinIR-M_noise15
    args['tile'] = None                  # help ='Tile size, None for no tile during testing (testing as a whole)'
    args['tile_overlap']= 32,            # help='Overlapping of different tiles'
    args['training_patch_size'] = 64     #  help='patch size used in training SwinIR. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from models.network_swinir import SwinIR as net
    # 001 classical image sr
    if args['task'] == 'classical_sr':
        model = net(upscale=args['scale'], in_chans=3, img_size=args['training_patch_size'], window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args['task'] == 'lightweight_sr':
        model = net(upscale=args['scale'], in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 004 grayscale image denoising
    elif args['task'] == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args['model_path'])
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
        
    model = model.to(device)
    
    return model

def load_model_RestormerDB(Pretrain):
    ''' Restormer for motion deblurring
    '''
    model_path = os.getcwd() + '/model_zoo/Restormer_motion_deblurring.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------
    # load model
    from models.restormer_arch import Restormer
    inp_channels=3 
    out_channels=3 
    dim = 48
    num_blocks = [4,6,6,8]
    num_refinement_blocks = 4
    heads = [1,2,4,8]
    ffn_expansion_factor = 2.66
    bias = False
    LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
    dual_pixel_task = False   
    
    model = Restormer(inp_channels=inp_channels, out_channels= out_channels, dim=dim, num_blocks=num_blocks,
                      num_refinement_blocks=num_refinement_blocks, heads = heads, ffn_expansion_factor=ffn_expansion_factor,
                      bias = bias, LayerNorm_type=LayerNorm_type,dual_pixel_task=dual_pixel_task )
    if(Pretrain):
        model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

if __name__ == '__main__':
    # read image
    import skimage
    import numpy as np
    import matplotlib.pyplot as plt
    data_fol = os.getcwd()
    img_gray = np.float32( skimage.io.imread(data_fol + '/cameraman.png')/255. )# 
    # img_gray = np.float32(skimage.color.rgb2gray(img_rgb))
    img = img_gray[::1,::1]
    
    #%% Test NAFNet denoising of color image
    model = load_model_KBNet()
    restored = np.zeros_like(img_gray)
    np.random.seed(seed=0)  # for reproducibility
    img += np.random.normal(0, 50. / 255., img.shape)
    plt.figure(1)
    plt.imshow(img)
    with torch.no_grad(): 
        img_n = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float()
        img_n = torch.clamp(img_n, 0 ,1.0)
        img_c = torch.clamp( model(img_n), 0, 1.0)
        # img_c = img_c - torch.min(img_c)
        # img_c = img_c/torch.max(img_c)
        img_c_cpu = img_c.detach().cpu().squeeze(0).squeeze(0).numpy()
    
    plt.figure(2)
    plt.imshow(img_c_cpu)
    img_c_cpu2 = 0.5 * img_c_cpu + (1-0.5)*img_n.squeeze(0).squeeze(0).cpu().numpy()
    
    plt.figure(3)
    plt.imshow(img_c_cpu2)

    
    
    