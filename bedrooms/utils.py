import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import numpy as np
from networks import NetType

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, netType: NetType, config, best_valid_loss=float('inf')
    ):
        if NetType.DISCRIMINATOR == netType:
            self.name = config['EXP_NAME'] + '_D'
        if NetType.GENERATOR == netType:
            self.name = config['EXP_NAME'] + '_G'
        self.models_path = config['PATHS']['MODELS']
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, netType: NetType, config
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, self.models_path + '/final_' + self.name + '.pth')

def save_model(model, optimizer, criterion, netType: NetType, config):
    """
    Function to save the trained model to disk.
    """
    if NetType.DISCRIMINATOR == netType:
        exp_name = config['EXP_NAME'] + '_D'
    if NetType.GENERATOR == netType:
        exp_name = config['EXP_NAME'] + '_G'
    models_path = config['PATHS']['MODELS']
    epochs = config['TRAIN']['NUM_EPOCHS']
    
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, models_path + '/final_' + exp_name + '.pth')

def save_plots(train_accs, train_losses, config):
    """
    Function to save the loss and accuracy plots to disk.
    """
    grapgs_path = config['PATHS']['GRAPHS']
    fig_size = (config['FIG_SIZE'][0], config['FIG_SIZE'][1])
    
    # accuracy plots
    exp_name = config['EXP_NAME']
    plt.figure(figsize=fig_size)
    for train_acc in train_accs:
        plt.plot(
            train_acc['accuracies'],
            linestyle='-', 
            label=train_acc['label']
        )
    plt.title('Accuracy ' + exp_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(grapgs_path + '/accuracy_' + exp_name + '.png')
    
    # loss plots
    plt.figure(figsize=fig_size)
    for train_loss in train_losses:
        plt.plot(
            train_loss['losses'],
            linestyle='-', 
            label=train_loss['label']
        )
    plt.title('Loss ' + exp_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(grapgs_path + '/loss_' + exp_name + '.png')

def plot_dataset_examples(dataloader, device, config):
    grapgs_path = config['PATHS']['GRAPHS']
    exp_name = config['EXP_NAME']
    real_batch = next(iter(dataloader))
    data =  np.transpose(
                vutils.make_grid(
                    real_batch[0].to(device)[:64], padding=2, normalize=True
                ).cpu(),
                (1, 2, 0),
            )
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    if config['VIZUALISE']:
        plt.imshow(data)
    else:
        plt.savefig(grapgs_path + '/dataset_examples_' + exp_name + '.png')

def plot_generated_examples(fake, config):
    grapgs_path = config['PATHS']['GRAPHS']
    exp_name = config['EXP_NAME']
    data = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0))
    
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images")
    if config['VIZUALISE']:
        plt.imshow(data)
    else:
        plt.savefig(grapgs_path + '/generated_examples_' + exp_name + '.png')

def plot_interpolated_examples(fake_interp, config):
    grapgs_path = config['PATHS']['GRAPHS']
    exp_name = config['EXP_NAME']
    data = np.transpose(
        vutils.make_grid(fake_interp, nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Fake Images Interpolated")
    if config['VIZUALISE']:
        plt.imshow(data)
    else:
        plt.savefig(grapgs_path + '/interpolated_images' + exp_name + '.png')
