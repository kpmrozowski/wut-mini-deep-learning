import torch.nn as nn
import torchvision
from enum import Enum

class NetType(Enum):
    GENERATOR = 1
    DISCRIMINATOR = 2
    ENCODER = 3
    DECODER = 4

class ModelName(Enum):
    DCGAN = 'DCGAN'
    DCGANProgressive = 'DCGANProgressive'
    VAE = 'VAE'

class OptimizerName(Enum):
    ADAM = 'Adam'
    ADADELTA = 'Adadelta'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GeneratorDCGAN(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(GeneratorDCGAN, self).__init__()
        self.net_type = NetType.GENERATOR
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, nc, ndf):
        super(DiscriminatorDCGAN, self).__init__()
        self.net_type = NetType.DISCRIMINATOR
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, input):
        return self.main(input)


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


class EncoderVAE(nn.Module):
    def __init__(self, nc, nz, ndf):
        super(EncoderVAE, self).__init__()
        self.net_type = NetType.ENCODER
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(ndf * 8 * 4 * 4, 2 * nz),
            nn.ReLU(True),
        )
    def forward(self, input):
        return self.main(input)


class DecoderVAE(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(DecoderVAE, self).__init__()
        self.net_type = NetType.DECODER
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (ngf * 8, 4, 4)),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)


