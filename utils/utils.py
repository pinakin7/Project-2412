from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

DATA_DIR = "../data"

num_channels = 1
image_size = 28


def plot_images_grid(num_images, img_size, image_tensors, labels, label):
    if image_tensors.shape != (num_images, num_channels, img_size, img_size):
        raise ValueError(f"Input tensor shape must be [{num_images}, num_channels, {img_size}, {img_size}].")

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


def show_images(images, title=""):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


def print_red(text):
    print("\033[91m{}\033[0m".format(text))


def index_to_onehot(index, n):
    if index.dim() == 1:
        index = index.unsqueeze(1)
    onehot = torch.zeros(index.size(0), n).to(index.device)
    onehot.scatter_(1, index, 1)
    return onehot


@dataclass
class GANHyperparameter:
    num_classes: int = 10
    batch_size: int = 128
    num_epochs: int = 50
    latent_size: int = 32
    n_critic: int = 5
    critic_size: int = 1024
    generator_size: int = 1024
    critic_hidden_size: int = 1024
    gp_lambda: float = 10.
    dp_sigma = {10: 1.1, 5: 1.35, 40: 0.75}
    dp_target_epsilons: tuple[int] = (5, 10, 40)


@dataclass
class VAEHyperparameter:
    num_classes: int = 10
    batch_size: int = 128
    encoder_layer_sizes: tuple[int] = (784, 256)
    latent_size: int = 2
    decoder_layer_sizes: tuple[int] = (256, 784)
    conditional: bool = True
    learning_rate: float = 0.001
    num_epochs: int = 40
    print_every: int = 200
    dp_learning_rate: float = 0.01
    dp_num_epochs: int = 300
    dp_print_every: int = 500
    dp_max_grad_norm: float = 1.0
    dp_target_epsilons: tuple[int] = (5, 10, 40)
    dp_target_delta: float = 0.00001
    dp_sigma = {5: 1.35, 10: 1.1, 40: 0.75}


@dataclass
class DiffusionHyperparameter:
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 0.0001
    n_steps: int = 1000
    min_beta: float = 0.001
    max_beta: float = 0.02
    grad_clamp: float = 0.5
    dp_learning_rate: float = 0.001
    np_noise: float = 0.00001
    dp_sigma = {5: 1.35, 10: 1.1, 40: 0.75}
    dp_target_epsilons: tuple[int] = (5, 10, 40)
    store_path:str = "models/diffusion"


def vae_loss_function(reconstructed_x, x, mean, log_var):
    binary_cross_entropy = torch.nn.functional.binary_cross_entropy(
        reconstructed_x.view(-1, image_size * image_size), x.view(-1, image_size * image_size), reduction='sum'
    )
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (binary_cross_entropy + kl_divergence) / x.size(0)


def show_forward_process(model, loader, device):
    for batch in loader:
        images = batch[0]

        show_images(images, "Original Images")

        for noise_percent in [0.25, 0.5, 0.75, 1.0]:
            show_images(model(images.to(device), [int(noise_percent * model.n_steps) - 1 for _ in range(len(images))]),
                        f"Noisy Images Percent: {noise_percent}%")
        break


def generate_new_images(model, num_samples=16, device=None, c=1, h=28,
                        w=25):
    with torch.no_grad():
        if device is None:
            device = model.device

        x = torch.randn(num_samples, c, h, w).to(device)

        for index, time in enumerate(list, range(model.n_steps))[::-1]:
            time_tensor = (torch.ones(num_samples, 1) * time).to(device).long()
            eta_theta = model.backward(x, time_tensor)

            alpha_t = model.alphas[time]
            alpha_t_bar = model.alpha_bars[time]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if time > 0:
                z = torch.randn(num_samples, c, h, w).to(device)

                # Option 1: When sigma_time^2 = beta_time
                beta_t = model.betas[time]
                sigma_t = beta_t.sqrt()

                # Option 2: Sigma_time^2 = beta_tilda_time
                # prev_alpha_t_bar = model.alpha_bars[time-1] if time > 0 else model.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                x = x + sigma_t * z
    return x.permute(0, 2, 3, 1)
