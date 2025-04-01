import os
import json, random
import sys

from .loss.loss_provider import LossProvider
from .util.ssim import *
import math
import lpips
import torch
import torch.nn as nn
import numpy as np
from .decoder_util import *

import torchvision.transforms as transforms
from pytorch_wavelets import DWTInverse, DWTForward

'''
Return Decoder Attributes

cfg_path : path to config JSON file.

TODO : make it work on settings other than 16-bit msgs.

'''

class DecoderAttributes:
    def __init__(self, 
                 cfg_path = './cfg.json', seed=42):
        
        try:
            with open(cfg_path, 'r') as f:
                self.cfg = json.load(f)
        except:
            raise FileNotFoundError
        
        print(os.getcwd())


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dec = torch.jit.load(self.cfg['decoder_path']).to(self.device)
        self.decoder_path = self.cfg['decoder_path']
        msg_str, self.msg = generate_and_pick_message(n_bits = self.cfg["n_bits"], seed=seed)
        os.makedirs("./decoder/generated_msgs", exist_ok = True)
        self.cfg['msg_save_path'] = self.cfg['msg_save_path'] + f"_{seed}.txt"
        # write key to the message file
        with open(self.cfg['msg_save_path'], 'w') as kf:
            kf.write(msg_str)
        
        self.message_path = self.cfg['msg_save_path']
        self.loss_dict = {}

        # losses

        # message loss (full-resolution)
        if self.cfg['loss_w'] == 'mse':
            self.loss_dict['loss_w'] = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2)
        elif self.cfg['loss_w'] == 'bce':
            self.loss_dict['loss_w'] = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
        else:
            raise NotImplementedError
        
        # image loss (full-resolution)

        provider = LossProvider()

        if self.cfg['loss_i'] == 'mse':
            self.loss_dict['loss_i'] = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
        elif self.cfg['loss_i'] == 'watson-dft':
            loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
            loss_percep = loss_percep.to(self.device)
            self.loss_dict['loss_i'] = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif self.cfg['loss_i'] == 'watson-vgg':
            loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
            loss_percep = loss_percep.to(self.device)
            self.loss_dict['loss_i'] = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif self.cfg['loss_i'] == 'ssim':
            loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
            loss_percep = loss_percep.to(self.device)
            self.loss_dict['loss_i'] = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        else:
            raise NotImplementedError

        # patch size
        self.patch_size = self.cfg['patch_size']

        # patchwise rgb loss (L1 loss - MAE)
        self.loss_dict['loss_rgb'] = nn.L1Loss(reduction='mean').to(self.device)
        # patchwise tv loss
        self.loss_dict['loss_tv'] = total_variation_loss
        # patchwise ssim loss
        provider_deferred = LossProvider()
        loss_percep_deferred = provider_deferred.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep_deferred = loss_percep_deferred.to(self.device)
        self.loss_dict['loss_ssim'] = lambda imgs_w, imgs: loss_percep_deferred((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]

        # trainable coef lambda_i for full resolution
        self.lambda_i = nn.Parameter(torch.tensor(self.cfg['lambda_i']))
        self.lambda_i_opt = torch.optim.Adam([{'params': self.lambda_i, "lr" : 0.0003}], betas=(0.9,0.99))

        # hyperparams
        self.lambda_l1 = self.cfg['lambda_l1']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_tv = self.cfg['lambda_tv']

        
        


