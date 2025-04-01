import torch
import lpips
from .util.ssim import ssim
import random

def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs

lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def ssim(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim(img1, img2, window_size, size_average)

def compute_lpips(img1, img2, net='alex', format='NCHW'):

    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == 'alex':
        return lpips_alex(img1, img2)
    elif net == 'vgg':
        return lpips_vgg(img1, img2)

def total_variation_loss(image):
    """
    계산한 TV (Total Variation) loss를 반환합니다.

    Parameters:
        image (torch.Tensor): 입력 이미지 텐서. shape는 [batch_size, num_channels, height, width]여야 합니다.

    Returns:
        torch.Tensor: TV loss 스칼라 텐서.
    """
    # 이미지 차원 추출
    batch_size, num_channels, height, width = image.size()

    # 이미지를 이용하여 수평 방향과 수직 방향으로 그래디언트 계산
    horizontal_grad = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    vertical_grad = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    # 그래디언트의 합 구하기
    tv_loss = torch.mean(horizontal_grad) + torch.mean(vertical_grad)

    return tv_loss


def generate_messages(start, end, nbits, fix_seed):
    '''
    - 시드를 설정해줌으로써 1000개의 key는 항상 같은 값으로 생성
    - 각 사용자는 자신의 범위에 해당하는 메시지만 저장
    '''

    random.seed(fix_seed) # 이미 전체 코드에서 선언되겠지만, 확실하게 하기 위해서 한번 더 선언한 것
    torch.manual_seed(fix_seed)

    # key 저장 리스트
    key_list = []
    
    # 1000개의 메시지를 생성하지만, 각 사용자는 자신의 범위에 해당하는 메시지만 저장
    
    torch.rand
    key_tensor = (torch.randn(1000, nbits) > 0).float() # 1000 nbits messages
    

    
    return key_tensor

'''
returns a message (n-bit) in a tuple of string and torch tensor
'''

def generate_and_pick_message(n_messages = 500, n_bits = 16, pick_idx = 0,seed=42, device = 'cuda'):
    
    msg_tensor = generate_messages(0, n_messages, n_bits, seed)
    msg = msg_tensor[pick_idx]
    #print(msg_str)
    msg_str = ''.join([str(int(t)) for t in msg])
    print(msg)
    return msg_str, msg.unsqueeze(dim=0).to(device)