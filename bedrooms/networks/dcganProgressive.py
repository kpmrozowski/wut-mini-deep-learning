import torch.nn as nn
import torchvision
from mytypes import NetType

class GeneratorDCGANProgressive(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(GeneratorDCGANProgressive, self).__init__()
        print(ngf)
        self.net_type = NetType.GENERATOR
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            ),
        ])
        self.finish = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 16, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ),
        ])
    def forward(self, input, n, alpha):
        input = self.blocks[0](input)
        for i in range(1, n):
            input = self.blocks[i](input)
        if n > 0:
            small = self.finish[n - 1](input)
            small_size = small.shape[2]
            small = torchvision.transforms.functional.resize(small, small_size * 2, torchvision.transforms.InterpolationMode.NEAREST)
            big = self.blocks[n](input)
            big = self.finish[n](big)
            input = small * (1 - alpha) + big * alpha
        else:
            input = self.finish[n](input)
        return input


class DiscriminatorDCGANProgressive(nn.Module):
    def __init__(self, nc, ndf):
        super(DiscriminatorDCGANProgressive, self).__init__()
        print(ndf)
        self.net_type = NetType.DISCRIMINATOR
        self.begin = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nc, ndf, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(nc, ndf * 2, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(nc, ndf * 4, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(nc, ndf * 8, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(nc, ndf * 16, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            ),
        ])
    def forward(self, input, n, alpha):
        if n > 0:
            small = self.begin[len(self.begin) - 1 - n + 1](input)
            small_size = small.shape[2]
            small = torchvision.transforms.functional.resize(small, small_size // 2, torchvision.transforms.InterpolationMode.NEAREST)
            big = self.begin[len(self.begin) - 1 - n](input)
            big = self.blocks[len(self.blocks) - 1 - n](big)
            input = small * (1 - alpha) + big * alpha
            pass
        else:
            input = self.begin[len(self.begin) - 1 - n](input)
        for i in range(1, n):
            input = self.blocks[len(self.blocks) - 1 - n + i](input)
        input = self.blocks[len(self.begin) - 1](input)
        return input
