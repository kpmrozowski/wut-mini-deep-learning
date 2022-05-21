# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
import os
directory_path = os.getcwd()
print("My current directory is : " + directory_path)

folder_name = os.path.basename(directory_path)

if 'bedrooms' != folder_name:
    print("Your directory name is : " + folder_name, 'but you should be in directory \"bedrooms\" containing folders \"dataset\" and \"configs\"')
    exit()

# %% [markdown]
# # Throat cleaning

# %%
# %matplotlib inline

# %%
from utils import SaveBestModel, save_model, save_plots, plot_dataset_examples, plot_generated_examples, plot_interpolated_examples
from networks import NetType, GeneratorDCGAN, DiscriminatorDCGAN, ModelName, OptimizerName

# %%
import random
import argparse
from tqdm.auto import tqdm
import yaml
from iglovikov_helper_functions.config_parsing.utils import object_from_dict

# import torch.backends.cudnn as cudnn
import torchvision
import torch

# %% [markdown]
# # Argument parsing

# %% 
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str,
    help='config for the experiment')
args = vars(parser.parse_args())
config_path = args['config']
if (not os.path.exists(config_path)):
    print('no such file:', config_path)
    exit()

with open(args['config']) as file:
   conf = yaml.safe_load(file)

if conf['DEBUG']:
   print(conf)

random.seed(conf['SEED'])
torch.manual_seed(conf['SEED'])

# %% [markdown]
# # Data loading

# %%
image_size = 64
dataset = torchvision.datasets.ImageFolder(
    root=conf['PATHS']['DATASET'],
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=conf['TRAIN']['BATCH_SIZE'],
    shuffle=conf['TRAIN']['SHUFFLE'],
    num_workers=conf['TRAIN']['WORKERS']
)

device_str = conf['TRAIN']['DEVICE']

# unspecified device
if 0 == len(device_str):
    device_str = "cuda:0" if (torch.cuda.is_available() and conf['TRAIN']['NGPU'] > 0) else "cpu"
# cuda choosed
elif 'cuda:0' == device_str and (not torch.cuda.is_available() or not conf['TRAIN']['NGPU'] > 0):
    device_str = 'cpu'

print('DEVICE:', device_str)
device = torch.device(device_str)

# %% [markdown]
# # Examples from dataset
# %%
plot_dataset_examples(dataloader, device=device, config=conf)

# %% [markdown]
# # Networks

# %%
if ModelName.DCGAN._value_ == conf['MODEL_NAME']:
    netG = GeneratorDCGAN(
        device=device,
        ngpu=conf['TRAIN']['NGPU'],
        nc=conf['DCGAN']['ARCHITECTURE']['NC'],
        nz=conf['NZ'],
        ngf=conf['DCGAN']['ARCHITECTURE']['NGF']
    ).to(device)
    netD = DiscriminatorDCGAN(
        device=device,
        ngpu=conf['TRAIN']['NGPU'],
        nc=conf['DCGAN']['ARCHITECTURE']['NC'],
        ndf=conf['DCGAN']['ARCHITECTURE']['NDF']
    ).to(device)
elif ModelName.VAE._value_ == conf['MODEL_NAME']:
    pass #TODO

criterion = object_from_dict(conf['TRAIN']['CRITERION'])
# torch.nn.BCELoss()

# %%
if conf['VERBOSE']:
    print(netG)
    print(netD)

# %% [markdown]
# # Training

# %%
adadelta = torch.optim.Adadelta(netG.parameters(),
                                lr=conf['ADADELTA']['LR'],
                                rho=conf['ADADELTA']['RHO'],
                                eps=conf['ADADELTA']['EPS'],
                                weight_decay=0.001)
adam = torch.optim.Adam(netD.parameters(),
                        lr=conf['ADAM']['LR'],
                        betas=(conf['ADAM']['BETA1'], 0.999))
if OptimizerName.ADADELTA._value_ == conf['OPTIMIZER_G_NAME']:
    optimizerG = adadelta
elif OptimizerName.ADAM._value_ == conf['OPTIMIZER_G_NAME']:
    optimizerG = adam
if OptimizerName.ADADELTA._value_ == conf['OPTIMIZER_D_NAME']:
    optimizerD = adadelta
elif OptimizerName.ADAM._value_ == conf['OPTIMIZER_D_NAME']:
    optimizerD = adam

saveBestModelG = SaveBestModel(NetType.GENERATOR, conf)
saveBestModelD = SaveBestModel(NetType.DISCRIMINATOR, conf)

real_label = 1.0
fake_label = 0.0

# %%
G_losses = []
D_losses = []
D_x_table = []
D_G_z1_table = []
D_G_z2_table = []

print("Starting Training Loop...")
for epoch in range(conf['TRAIN']['NUM_EPOCHS']):
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):

        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, conf['NZ'], 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

    print(
        "[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
        % (
            epoch + 1,
            conf['TRAIN']['NUM_EPOCHS'],
            errD.item(),
            errG.item(),
            D_x,
            D_G_z1,
            D_G_z2,
        )
    )

    D_losses.append(errD.item())
    G_losses.append(errG.item())
    D_x_table.append(D_x)
    D_G_z1_table.append(D_G_z1)
    D_G_z2_table.append(D_G_z2)
    if (0.50 < D_x < 0.99):
        saveBestModelD(errD.item(), epoch, netD, optimizerD, criterion)
    if (0.01 < D_G_z2 < 0.50):
        saveBestModelG(errG.item(), epoch, netG, optimizerG, criterion)

# %% [markdown]
# # Results

# %%
save_model(netD, optimizerD, criterion, NetType.DISCRIMINATOR, conf)
save_model(netG, optimizerG, criterion, NetType.GENERATOR, conf)
save_plots(
    train_accs=[
        {'label': 'D(x)', 'accuracies': D_x_table},
        {'label': 'D(G(z))[fake]', 'accuracies': D_G_z1_table},
        {'label': 'D(G(z))[real]', 'accuracies': D_G_z2_table}],
    train_losses=[
        {'label': 'Generator', 'losses': D_x_table},
        {'label': 'Discriminator', 'losses': D_G_z1_table}],
    config=conf)

# %%
fixed_noise = torch.randn(64, conf['NZ'], 1, 1, device=device)
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
plot_generated_examples(fake=fake, config=conf)

# %%
interp_noise = (
    (torch.tensor(list(range(0, 10, 1))) / 9).reshape(10, 1, 1, 1).tile(1, conf['NZ'], 1, 1)
)
with torch.no_grad():
    interp = netG(interp_noise).detach().cpu()
plot_interpolated_examples(fake_interp=interp, config=conf)

# %%
