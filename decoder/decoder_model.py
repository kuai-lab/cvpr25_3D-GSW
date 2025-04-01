import torch
import torch.nn as nn
from torchvision import models

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
    
class Classifier(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, classifier_channels,classifier_blocks):
        super(Classifier, self).__init__()

        layers = [ConvBNRelu(3, classifier_channels)]
        for _ in range(classifier_blocks-1):
            layers.append(ConvBNRelu(classifier_channels, classifier_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(classifier_channels, 1)

    def forward(self, image):
        x = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x = x.squeeze(3).squeeze(2)
        x = self.linear(x)
        # X = torch.sigmoid(X)
        return torch.sigmoid(x)
    
class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(Decoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w, c):
        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) * c # b d
        return x                      