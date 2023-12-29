import torch
import torch.nn as nn
from torchvision.utils import save_image

from ModelArchitectures.Diffusion.Diffusion import Diffusion
from ModelArchitectures.Diffusion.UNet import UNet
from utils.utils import DiffusionHyperparameter, generate_new_images
from tqdm import tqdm


def train(loader, optim, device, dp=False, epsilon=None, display=False):
    hp = DiffusionHyperparameter()
    model = Diffusion(UNet(hp.n_steps), hp.n_steps, hp.min_beta, hp.max_beta, device)

    if dp:
        for param in model.parameters():
            param.requires_grad = True
            param.register_hook(
                lambda grad: grad + (1 / hp.batch_size) * hp.dp_sigma[epsilon] * torch.randn(param.shape).to(device)
            )
            model.network.time_embed.requires_grad = False
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = model.n_steps

    for epoch in tqdm(range(hp.num_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(
                tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{hp.num_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = model(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = model.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()

            # weight clipping for privacy guarantee
            if dp:
                for param in model.network.parameters():
                    param.data.clamp_(-hp.grad_clamp, hp.grad_clamp)
                    epoch_loss += loss.item() * len(x0) / len(loader.dataset)

            loss.backward()
            optim.step()

        # Display images generated at this epoch
        if display and epoch % 10 == 0:
            generated = generate_new_images(model, device=device, n_samples=100)
            save_image(generated, f"Generated Image {epoch}.png")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"{hp.store_path}_{epsilon}.pth")
            log_string += " --> Best model ever (stored)"

        print(log_string)
