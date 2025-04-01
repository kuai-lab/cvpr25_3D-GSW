import numpy
import sys
from os import listdir
from PIL import Image

from utils.general_utils import safe_state
from gaussian_renderer import render
from argparse import ArgumentParser, Namespace
from decoder.init_decoder import DecoderAttributes
from decoder.util.ssim import ssim
from pytorch_wavelets import DWTInverse, DWTForward
from scene import Scene, GaussianModel
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from arguments import ModelParams, PipelineParams, OptimizationParams
import lpips
import os
'''
bit msgs expected as inputs 
'''

lpips_net = lpips.LPIPS(net='vgg').cuda()


def bit_acc(gt, pred):
    same = ~torch.logical_xor(gt>0, pred>0)
    bit_accs = torch.sum(same)/same.shape[-1]
    return bit_accs

def compute_ssim(gt, pred, window_size=11, size_average=True, format='NCHW'):
    return ssim(gt, pred, window_size, size_average)

def compute_psnr(gt, pred):
    mse_loss = F.mse_loss(gt, pred)
    return -10. * torch.log10(mse_loss)


def compute_lpips(gt, pred): # so awfully slow
    #gt, pred = gt.detach().cpu().clone(), pred.detach().cpu().clone()
    torch.cuda.empty_cache()
    return lpips_net(gt, pred)

    
if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--experiment_path", type=str, default = None)
    parser.add_argument("--gt_image_path", type=str, default = None)

    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--decoder_att", default="", type=str)
    args = parser.parse_args(sys.argv[1:])

    dec_attrs = DecoderAttributes(cfg_path=args.decoder_att, seed= args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading generated key.. ')
    with open(dec_attrs.message_path, 'r') as f:
        key = f.read()
        keyarr = [int(_) for _ in key]
        keytensor = torch.tensor(keyarr, dtype=torch.float32).to(device)[None, :]
    
    totensor = torchvision.transforms.ToTensor()


    print(args.experiment_path)
    wm_img_path = os.path.join(args.experiment_path, 'test', f'ours_{args.epoch}', 'renders/')

    with open(f'{args.experiment_path}/log_epoch_{args.epoch}.txt', 'w') as log_file:
        log_file.write(f"Loading Test Cameras.. \n")
        
        avg_test_bitacc = 0.
        avg_test_psnr = 0.
        avg_test_ssim = 0.
        avg_test_lpips = 0.
        num = 0

        for file in listdir(wm_img_path):   
            num+=1 
            image_dir = wm_img_path + file
            gt_image_dir = args.gt_image_path + file
            img = Image.open(image_dir)
            gt_img = Image.open(gt_image_dir)

            image_wm = totensor(img).unsqueeze(0).contiguous().to(device)
            image_gt = totensor(gt_img).unsqueeze(0).contiguous().to(device)

            decoder = dec_attrs.dec
            LL_img, _ = DWTForward(wave='bior4.4', J=2, mode='periodization').to(device)(image_wm)
            # decode the message from the image
            pred_msg = decoder(LL_img)

            bit_accu = bit_acc(keytensor, pred_msg)
            psnr = compute_psnr(image_gt, image_wm)
            ssim_ = compute_ssim(image_gt, image_wm)
            lpips_ = compute_lpips(image_gt, image_wm)


            avg_test_bitacc += bit_accu.item()
            avg_test_psnr += psnr.item()
            avg_test_ssim += ssim_.item()
            avg_test_lpips += lpips_.item()

        log_file.write('Test cameras results : \n')
        log_file.write(f'bit accuracy : {round(avg_test_bitacc / num, 5)}\n')
        log_file.write(f'ssim : {round(avg_test_ssim / num, 5)}\n')
        log_file.write(f'psnr : {round(avg_test_psnr / num, 5)}\n')
        log_file.write(f'lpips : {round(avg_test_lpips / num, 5)}\n')
