import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

img_height = 256
img_width = 256
channels = 3

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()
        
        channels = input_shape[0]
        
        # Initial Convolution Block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Output Layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()
                 ]
        
        # Unpacking
        self.model = nn.Sequential(*model) 

    def forward(self, x):
        return self.model(x)

input_shape = (channels, img_height, img_width) # (3,256,256)
n_residual_blocks = 9 # suggested default, number of residual blocks in generator

G_AB = GeneratorResNet(input_shape, n_residual_blocks)

G_AB.load_state_dict(torch.load("1000.G_BA.model", map_location="cpu"))

cuda = torch.cuda.is_available()

if cuda:
    G_AB = G_AB.cuda()

from PIL import Image
import torchvision.transforms as transforms

transforms_ = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
transform = transforms.Compose(transforms_)

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

import matplotlib.pyplot as plt

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def sample_images(imgs):
    """show a generated sample from the test set"""
    G_AB.eval()
    real_A = imgs.type(Tensor) # A : monet
    fake_B = G_AB(real_A).detach()
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=8, normalize=True)
    fake_B = make_grid(fake_B, nrow=8, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B), 1)

    save_image(image_grid.cpu(), 'test_sample.png')

def center_crop_to_min_dimension(image):
    width, height = image.size
    min_size = min(width, height)
    left = (width - min_size) / 2
    top = (height - min_size) / 2
    right = (width + min_size) / 2
    bottom = (height + min_size) / 2
    return image.crop((left, top, right, bottom))

imgs = []
for i in range(1, 9):
    f = "test%d.JPG" % i
    imgs.append(center_crop_to_min_dimension(Image.open(f)))
imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], 0)
print(imgs.shape)
sample_images(imgs)
