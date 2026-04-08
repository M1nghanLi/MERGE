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


    