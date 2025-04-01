import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional
import cv2
import numpy as np
from PIL import Image
import random

interpolation = functional.InterpolationMode('bilinear')
totensor = torchvision.transforms.ToTensor()

class GaussianBlurAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 3
        self.sigma = 0.1
    
    def forward(self, image):
        blurred_img = functional.gaussian_blur(image, self.kernel_size, self.sigma)
        return blurred_img

class RotationAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = 30
    def forward(self, image):
        rand_num = random.uniform(-1, 1)
        rotated_img = functional.rotate(image, self.angle*rand_num)#, expand = True)
        return rotated_img
    
class CropAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 0.5 
    
    def forward(self, image):
        scale = np.sqrt(self.scale)
        edges_size = [int(s* scale) for s in image.size][::-1]
        cropped_img = functional.center_crop(image, edges_size)
        return cropped_img

class ResizeAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 0.75
    def forward(self, image):
        scale = np.sqrt(self.scale)
        edges_size = [int(s*scale) for s in image.size][::-1]
        resized_img = functional.resize(image, edges_size,interpolation=interpolation, antialias=True)
        return resized_img

class NoiseAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = 0.1
        
    def forward(self, image):
        tensor_img = totensor(image)
        noised_image = tensor_img + (self.sigma **2) * torch.randn_like(tensor_img)
        tensor_to_PIL = torchvision.transforms.ToPILImage()

        return tensor_to_PIL(noised_image)
    
class NoiseAttack_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = 0.05
        
    def forward(self, image):
        tensor_img = totensor(image)
        noised_image = tensor_img + (self.sigma **2) * torch.randn_like(tensor_img)
        tensor_to_PIL = torchvision.transforms.ToPILImage()

        return tensor_to_PIL(noised_image)
    
class BrightnessAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 2.0 
    
    def forward(self, image):
        brightened_img = functional.adjust_brightness(image, self.factor)
        return brightened_img

class JPEGCompressAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.magnitude = 10
        # CopyRNeRF value


    def forward(self, image):
        numpy_img = np.array(image)
        cv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.magnitude]
        result, encimg = cv2.imencode('.jpg', cv_img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
        return decimg


class Attacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_layers = []
        self.noise_layers.append(GaussianBlurAttack())
        self.noise_layers.append(RotationAttack())
        self.noise_layers.append(CropAttack())
        self.noise_layers.append(ResizeAttack())
        self.noise_layers.append(NoiseAttack())
        self.noise_layers.append(BrightnessAttack())
        self.noise_layers.append(JPEGCompressAttack())
        self.noise_layers.append(NoiseAttack_2())


    def forward(self, image, idx):
        # random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        random_noise_layer = self.noise_layers[idx]
        return random_noise_layer(image)
        