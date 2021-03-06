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
    StyleGAN3 = 'StyleGAN3'

class OptimizerName(Enum):
    ADAM = 'Adam'
    ADADELTA = 'Adadelta'
