#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import random
import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.densify_high_frequency_dft import densify_high_frequency
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from decoder.init_decoder import DecoderAttributes
from pytorch_wavelets import DWTInverse, DWTForward
from torchviz import make_dot
import math
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View
from lpipsPyTorch import lpips
from vgg_loss import VGGLoss
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs

def compute_left_up(z, fovx, fovy):
    left = z*math.tan(fovx/2)
    up = z*math.tan(fovy/2)
    return left, up

def total_variation_loss(image):
 
    batch_size, num_channels, height, width = image.size()

    horizontal_grad = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    vertical_grad = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    tv_loss = torch.mean(horizontal_grad) + torch.mean(vertical_grad)

    return tv_loss

def prepare_output_and_logger(args):    
    
    args.exp_name = os.path.join("./output/", args.exp_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.exp_name))
    os.makedirs(args.exp_name, exist_ok = True)
    with open(os.path.join(args.exp_name, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, saving_epochs, checkpoint_iterations, checkpoint, debug_from, seed, device=device):
    first_epoch = 0
    
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians, load_iteration=-1)
    gaussians.training_setup(opt) 
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    dec_attrs = DecoderAttributes(args.decoder_att, seed)
    print(f'\n>>> Loading decoder from {dec_attrs.decoder_path}...')

    msg_decoder = dec_attrs.dec
    msg_decoder.eval()
    
    loss_type = dec_attrs.loss_dict

    # target message
    gt_msg = dec_attrs.msg
    
    # Find high frequency and split gaussians
    print("FGD=================================")
    orgin_gs_num = gaussians._xyz.shape[0]
    start = time.time()
    scene, gaussians = densify_high_frequency(scene, gaussians, dataset, opt, pipe)
    end = time.time()
    fgd_gs_num = gaussians._xyz.shape[0]
    print("fgd time : ", (end - start))
    print("Before FGD Gaussian num : ", orgin_gs_num)
    print("After FGD Gaussian num : ", fgd_gs_num)
    print("After FGD Gaussian num change : ", orgin_gs_num - fgd_gs_num)
    
    first_epoch += 1

    gstep = 0
    progress_bar = tqdm(range(0, args.epochs), total = args.epochs)
    print("tuning epoch : ", args.epochs)
    alpha = opt.alpha

    w1 = (torch.abs(gaussians._features_dc))**alpha
    w2 = (torch.abs(gaussians._features_rest))**alpha
    w3 = (torch.abs(gaussians._opacity))**alpha
    w4 = (torch.abs(gaussians._rotation))**alpha
    w5 = (torch.abs(gaussians.get_scaling))**alpha
    # w6 = (torch.abs(gaussians._xyz))**2.

    w1_rep = 1. / torch.exp(w1)
    w2_rep = 1. / torch.exp(w2)
    w3_rep = 1. / torch.exp(w3)
    w4_rep = 1. / torch.exp(w4)
    w5_rep = 1. / torch.exp(w5)
    # w6_rep = 1. / torch.exp(w6)
    
    msk1 = w1_rep / torch.sum(w1_rep, dim = 0) 
    msk2 = w2_rep / torch.sum(w2_rep, dim = 0)
    msk3 = w3_rep / torch.sum(w3_rep, dim = 0)
    msk4 = w4_rep / torch.sum(w4_rep, dim = 0) 
    msk5 = w5_rep / torch.sum(w5_rep, dim = 0)
    # msk6 = w6_rep / torch.sum(w6_rep, dim = 0)

    torch.cuda.empty_cache()
    
    
    for epoch_id in progress_bar: 
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        num_iteration =len(viewpoint_stack)
        print(f'number of cameras : {num_iteration}')

        log_loss = 0.0
        log_psnr = 0.0
        log_bit_acc = 0.0
        log_ssim = 0.0
        log_lpips = 0.0

        for iter_id in range(len(viewpoint_stack)):    
            gstep += 1
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            bg = torch.rand((3), device=device) if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            
            
            gt_image = viewpoint_cam.original_image.to(device)

            image = image.unsqueeze(0).contiguous()
            gt_image = gt_image.unsqueeze(0).contiguous().to(device)

            LL_img, yh = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)(image)
            LL_img_gt, yh_gt = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)(gt_image)
       
            # Extract watermark
            decoded = msg_decoder(LL_img) # b c h w -> b k

            loss_wm = loss_type['loss_w'](decoded, gt_msg)

            loss_im_mse = F.mse_loss(gt_image, image)
            
            psnr = -10.0 * math.log10(loss_im_mse)
            Ll1 = l1_loss(image, gt_image)
            loss_lpips = lpips(image, gt_image, net_type='vgg')

            # Wavelet subband Loss

            # LH
            lh_2 = yh[1][:,:,0,:,:]
            lh_1 = yh[0][:,:,0,:,:]
            lh_gt_2 = yh_gt[1][:,:,0,:,:]
            lh_gt_1 = yh_gt[0][:,:,0,:,:]


            # HL
            hl_2 = yh[1][:,:,1,:,:]
            hl_1 = yh[0][:,:,1,:,:]
            hl_gt_2 = yh_gt[1][:,:,1,:,:]
            hl_gt_1 = yh_gt[0][:,:,1,:,:]

            # HH
            hh_2 = yh[1][:,:,2,:,:]
            hh_1 = yh[0][:,:,2,:,:]
            hh_gt_2 = yh_gt[1][:,:,2,:,:]
            hh_gt_1 = yh_gt[0][:,:,2,:,:]


            loss_lhhlhh_mse =   torch.mean(torch.abs(lh_2 - lh_gt_2)) +  torch.mean(torch.abs(lh_1 - lh_gt_1)) + \
                                torch.mean(torch.abs(hl_2 - hl_gt_2)) + torch.mean(torch.abs(hl_1 - hl_gt_1)) + \
                                torch.mean(torch.abs(hh_2 - hh_gt_2)) +  torch.mean(torch.abs(hh_1 - hh_gt_1))

            loss =  opt.lambda_lpips * loss_lpips  + \
                    opt.lambda_i * Ll1 + \
                    opt.lambda_subband * loss_lhhlhh_mse + \
                    opt.lambda_wm * loss_wm 
            
            loss_dict = {}
            loss_dict['watermark_loss'] = loss_wm.detach().item()
            
            loss_dict['psnr'] = psnr
            loss_dict['ssim'] = ssim(image.squeeze(0).permute(1,2,0), gt_image.squeeze(0).permute(1,2,0)).item()
            loss_dict['lpips'] = lpips(image, gt_image, net_type='vgg').item()
        
            loss_dict['bit-accuracy'] = bit_acc(decoded, gt_msg).item()
            loss_dict['total_loss'] = loss.detach().item()
                        
            loss.backward()
            
            gaussians._features_dc.grad = gaussians._features_dc.grad * msk1
            gaussians._features_rest.grad = gaussians._features_rest.grad * msk2
            gaussians._opacity.grad = gaussians._opacity.grad * msk3
            gaussians._rotation.grad = gaussians._rotation.grad * msk4
            gaussians._scaling.grad = gaussians._scaling.grad * msk5
            # gaussians._xyz.grad = gaussians._xyz.grad * msk6

            log_loss += loss_dict['total_loss']
            log_psnr += loss_dict['psnr']
            log_bit_acc += loss_dict['bit-accuracy']
            log_ssim += loss_dict['ssim']
            log_lpips += loss_dict['lpips']

            torch.cuda.empty_cache()

            print()
            
            progress_bar.set_description(f'global-iteration-{gstep} | epoch-{epoch_id+1} :  \
                                 loss_wm={loss_dict["watermark_loss"]:.4f} bit-accuracy={loss_dict["bit-accuracy"]:.4f} psnr={loss_dict["psnr"]:.4f} \
                                 ssim={loss_dict["ssim"]:.4f} lpips={loss_dict["lpips"]:.4f}')
            
            
            gaussians.update_learning_rate(gstep)
            
            ####################################################################################################
            with torch.no_grad():

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        with torch.no_grad():
            if (epoch_id+1 in saving_epochs):
                print("\n[ITER {}] Saving Gaussians".format(epoch_id+1))
                scene.finetuning_save(epoch_id+1)
                
            loss_dict_test = {"psnr":0.0, "ssim":0.0, "lpips":0.0,"bit-accuracy":0.0}
            # Progress bar
            test_viewpoint_stack = scene.getTestCameras()
            test_viewpoint_cam = test_viewpoint_stack
            for idx, view in enumerate(test_viewpoint_cam[:3]):
                test_renderd_image = render(view, gaussians, pipe, bg)["render"]
                test_gt = view.original_image[0:3, :, :]
                test_image = test_renderd_image.unsqueeze(0).contiguous().to(device)
                test_gt_image = test_gt.unsqueeze(0).contiguous().to(device)

                LL_img, _ = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)(test_image)
                decoded = msg_decoder(LL_img)

                loss_im_mse = F.mse_loss(test_gt_image, test_image)
                psnr = -10.0 * math.log10(loss_im_mse)

                loss_dict_test['psnr'] += psnr
                loss_dict_test['ssim'] += ssim(test_image.squeeze(0).permute(1,2,0), test_gt_image.squeeze(0).permute(1,2,0)).item()
                loss_dict_test['lpips'] += lpips(test_image, test_gt_image, net_type='vgg').item()
                loss_dict_test['bit-accuracy'] += bit_acc(decoded, gt_msg).item()

            for stat_name in loss_dict_test:
                loss_dict_test[stat_name] /= len(test_viewpoint_cam[:3])
            print('<validation stats>\n')
            for stat_name in loss_dict_test:
                print(f'{stat_name} : {loss_dict_test[stat_name]:.4f}')
            print()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[1,2,3,4, 5,6, 7,8,9,10, 20, 30])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--msg_decoder_path", type=str, default = "./decoder/pretrained_decoder/16bits/16_256_checkpoint_whit.pth")
    parser.add_argument("--seed", type=int, default = 42)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # seed_everything(args.seed)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed_everything(args.seed)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.save_epochs, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.seed)

    # All done
    print("\nTraining complete.")