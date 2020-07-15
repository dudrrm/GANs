# input 256 x 256

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

# what is ngf? number of generator features?

# ================================================================== #
#                     Generator with skips                           #
# ================================================================== #

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

    # def encoder(self):
        # Dimensional changes of inputs
        # nc x 256 x 256
        # self.layer1 = nn.Sequential(
        #     nn.SpartialConvolution(input_num_channel, ngf, 4, 4, 2, 2, 1, 1),
        #     nn.LeakyReLU(0.2,true))
        self.en_layer1 = nn.Sequential(
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf x 128 x 128
        self.en_layer2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 2 x 64 x 64
        self.en_layer3 = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 4 x 32 x 32
        self.en_layer4 = nn.Sequential(
             nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=5, stride=1, padding=2),
             nn.BatchNorm2d(ngf*8),
             nn.LeakyReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 16 x 16
        self.en_layer5 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 8 x 8
        self.en_layer6 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 4 x 4
        self.en_layer7 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 2 x 2
        self.en_layer8 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

    # def decoder(self):
        # ngf * 8 x 1 x 1 ... check the batch_size
        self.de_layer1 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.Dropout(),
            nn.ReLU())
        # ngf * 8 x 2 x 2
        self.de_layer2 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.Dropout(),
            nn.ReLU())
        # ngf * 8 x 4 x 4
        self.de_layer3 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.Dropout(),
            nn.ReLU())
        # ngf * 8 x 16 x 16
        self.de_layer4 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU())
        # ngf * 4 x 32 x 32
        self.de_layer5 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU())
        # ngf * 2 x 64 x 64
        self.de_layer6 = nn.Sequential(
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU())
        # ngf x 128 x 128
        self.de_layer7 = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ngf),
            nn.ReLU())
        # nc x 256 x 256
        self.de_layer8 = nn.Sequential(
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=5, stride=1, padding=2),
            nn.Tanh())

    def forward(self, x):
        out = self.en_leyer1(x)
        out = self.en_layer2(out)
        out = self.en_layer3(out)
        out = self.en_layer4(out)
        out = self.en_layer5(out)
        out = self.en_layer6(out)
        out = self.en_layer7(out)
        out = self.en_layer8(out)

        # out = out.reshape(out.size(0), -1)

        out = self.de_layer1(out)
        out = self.de_layer2(out)
        out = self.de_layer3(out)
        out = self.de_layer4(out)
        out = self.de_layer5(out)
        out = self.de_layer6(out)
        out = self.de_layer7(out)
        out = self.de_layer8(out)

        return out

class G_unet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(G_unet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

    # def encoder(self):
        # Dimensional changes of inputs
        # nc x 256 x 256
        self.en_layer1 = nn.Sequential(
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf x 128 x 128
        self.en_layer2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 2 x 64 x 64
        self.en_layer3 = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 4 x 32 x 32
        self.en_layer4 = nn.Sequential(
             nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=5, stride=1, padding=2),
             nn.BatchNorm2d(self.ngf*8),
             nn.LeakyReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 16 x 16
        self.en_layer5 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 8 x 8
        self.en_layer6 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 4 x 4
        self.en_layer7 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # ngf * 8 x 2 x 2
        self.en_layer8 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

    # def decoder(self, output_nc, ngf):
        # ngf * 8 x 1 x 1 ... check the batch_size
        self.de_layer1 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.Dropout(),
            nn.ReLU())
        # ngf * 8 x 2 x 2
        self.de_layer2 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.Dropout(),
            nn.ReLU())
        # ngf * 8 x 4 x 4
        self.de_layer3 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.Dropout(),
            nn.ReLU())
        # ngf * 8 x 16 x 16
        self.de_layer4 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU())
        # self.ngf * 4 x 32 x 32
        self.de_layer5 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU())
        # self.ngf * 2 x 64 x 64
        self.de_layer6 = nn.Sequential(
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU())
        # self.ngf x 128 x 128
        self.de_layer7 = nn.Sequential(
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU())
        # nc x 256 x 256
        self.de_layer8 = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.output_nc, kernel_size=5, stride=1, padding=2),
            nn.Tanh())

    def forward(self, x):
        # encode x
        out = self.en_leyer1(x)
        out = self.en_layer2(out)
        out = self.en_layer3(out)
        out = self.en_layer4(out)
        out = self.en_layer5(out)
        out = self.en_layer6(out)
        out = self.en_layer7(out)
        out = self.en_layer8(out)
        # out = out.reshape(out.size(0), -1)
        # decode latent output
        out = self.de_layer1(out)
        out = self.de_layer2(out)
        out = self.de_layer3(out)
        out = self.de_layer4(out)
        out = self.de_layer5(out)
        out = self.de_layer6(out)
        out = self.de_layer7(out)
        out = self.de_layer8(out)

        return out


# ================================================================== #
#                   Markovian Discriminator                          #
# ================================================================== #

class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(Discriminator, self).__init__()

        # nc x 256 x 256
        self.layer1 = nn.Sequential(
                    nn.Conv2d(input_nc + output_nc, ndf, kernel_size=5, stride=1, padding=2),
                    nn.LeakyReLU(0.2))
        # ndf x 256 x 256
        self.layer2 = nn.Sequential(
                    nn.Conv2d(ndf, ndf*2, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(ndf*2),
                    nn.LeakyReLU(0.2))
        # ndf*2 x 256 x 256
        self.layer3 = nn.Sequential(
                    nn.Conv2d(ndf*2, 1, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid())
        # 1 x 256 x 256

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
