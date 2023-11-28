import matplotlib.pyplot as plt
import numpy as np
import torch

DATA_DIR = "../data"

label_class = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_channels = 3

# Attack model parameters
attack_noise_dim = 100
num_attack_generator_filter = 64
num_attack_discriminator_filter = 64
attack_batch_size = 100
attack_epochs = 20
attack_gen_lr = 0.00558
attack_disc_lr = 0.003
attack_img_size = 64

ATTACK_MODEL_PATH = "../models/attack/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = 1

# Attack model parameters
dpgan_noise_dim = 100
num_dpgan_generator_filter = 64
num_dpgan_discriminator_filter = 64
dogan_batch_size = 100
dpgan_epochs = 20
dpgan_gen_lr = 0.00558
dpgan_disc_lr = 0.003
dpgan_img_size = 32
DPGAN_MODEL_PATH = "../models/dpgan/"
GAN_MODEL_PATH = "../models/gan/"


def init_params_attack_model(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def init_params_dpgan_model(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.05)
        model.bias.data.fill_(0)
    elif classname.find('GroupNorm') != -1:
        model.weight.data.normal_(1.0, 0.05)
        model.bias.data.fill_(0)


def plot_images_grid(num_images, img_size, image_tensors, labels, label):
    if image_tensors.shape != (num_images, 3, img_size, img_size):
        raise ValueError(f"Input tensor shape must be [{num_images}, 3, {img_size}, {img_size}].")

    nrows = int(np.sqrt(num_images))
    ncols = num_images // nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_images):
        row, col = divmod(i, nrows)
        img = image_tensors[i]
        img = img.numpy()
        img = img / 2 + 0.5
        ax = axes[row, col]
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(label[labels[i]])
        ax.axis('off')
    plt.show()


def print_red(text):
    print("\033[91m{}\033[0m".format(text))
