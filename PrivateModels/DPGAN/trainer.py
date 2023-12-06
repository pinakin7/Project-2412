import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vsutils
from opacus import PrivacyEngine
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from MIA import mia
from DatasetGenerator.main import generate_dataset
from Discriminator import Discriminator
from Generator import Generator
from utils import utils


def train(run):

    # torch.use_deterministic_algorithms(True)  # for reproducible results

    data_loader = generate_dataset(utils.DATA_DIR, batch_size=utils.dpgan_batch_size, img_size=utils.dpgan_img_size,
                                   num_workers=0)
    gen_net = Generator(utils.dpgan_noise_dim, utils.num_dpgan_generator_filter, utils.num_channels,
                        utils.num_gpus).to(utils.device)
    gen_net.load_state_dict(torch.load("models/dpgan/Generator.pth"))

    disc_net = Discriminator(utils.num_channels, utils.num_dpgan_discriminator_filter, utils.num_gpus).to(utils.device)
    disc_net.apply(utils.init_params_dpgan_model)

    criterion = nn.BCELoss()

    optimizer_gen = optim.SGD(gen_net.parameters(), lr=utils.dpgan_gen_lr, momentum=0.5)

    optimizer_disc = optim.SGD(disc_net.parameters(), lr=utils.dpgan_disc_lr, momentum=0.9)


    real_label = 0.9
    fake_label = 0.1


    disc_privacy_engine = PrivacyEngine()
    disc_net, optimizer_disc, data_loader = disc_privacy_engine.make_private_with_epsilon(
        module=disc_net,
        data_loader=data_loader,
        optimizer=optimizer_disc,
        # noise_multiplier=utils.dpgan_discriminator_noise,
        max_grad_norm=utils.dpgan_discriminator_max_grad_norm,
        batch_first=True,
        poisson_sampling=False,
        target_epsilon=5.0,
        target_delta=utils.dpgan_generator_delta,
        epochs=utils.dpgan_epochs
    )
    # gen_privacy_engine = PrivacyEngine(accountant="gdp")
    # gen_net, optimizer_gen, data_loader = gen_privacy_engine.make_private(
    #     module=gen_net,
    #     optimizer=optimizer_gen,
    #     data_loader=data_loader,
    #     noise_multiplier=utils.dpgan_generator_delta,
    #     max_grad_norm=utils.dpgan_generator_max_grad_norm,
    #     batch_first=True
    # )

    run.watch(gen_net, log='all', log_freq=100)
    run.watch(disc_net, log='all', log_freq=100)

    for epoch in range(utils.dpgan_epochs):
        gen_net.train()
        disc_net.train()
        error_D, error_G, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0
        for i, data in tqdm(enumerate(data_loader, 0), desc=f"Epoch {epoch + 1}/{utils.dpgan_epochs}:",
                            total=len(data_loader)):
            ########################################################
            # (Step 1): Updating the Discriminator model:
            # # maximize log(D(x)) + log(1 - D(G(Z)))
            ########################################################
            data = data[0]
            batch_size = data.size(0)
            # training with the real data
            optimizer_disc.zero_grad(set_to_none=True)
            data = data.to(utils.device)
            label = torch.full((batch_size,), real_label, device=utils.device, dtype=torch.float32)

            output = disc_net(data).view(-1)
            err_disc_real = criterion(output, label)
            err_disc_real.backward()
            D_x = output.mean().item()

            # training with fake data
            noise = torch.randn(batch_size, utils.dpgan_noise_dim, 1, 1, device=utils.device)
            fake = gen_net(noise)
            label.fill_(fake_label)
            output = disc_net(fake.detach()).view(-1)
            err_disc_fake = criterion(output, label)
            err_disc_fake.backward()
            D_G_z1 = output.mean().item()
            error_D += err_disc_fake + err_disc_real
            nn.utils.clip_grad_norm(disc_net.parameters(), utils.dpgan_discriminator_max_grad_norm)
            optimizer_disc.step()


            ########################################################
            # (Step 2): Updating the Generator model:
            # # maximize log(D(G(Z)))
            ########################################################
            optimizer_gen.zero_grad(set_to_none=True)
            label.fill_(real_label)
            output = disc_net(fake).view(-1)
            err_gen = criterion(output, label)
            err_gen.backward()
            D_G_z2 = output.mean().item()
            optimizer_gen.step()
            error_G += err_gen.item()
            # scheduler.step(err_gen)

            if i % 100 == 0:
                with torch.no_grad():
                    fixed_noise = torch.randn(utils.dpgan_batch_size, utils.dpgan_noise_dim, 1, 1, device=utils.device)
                    fake = gen_net(fixed_noise).detach()
                    run.log({"image": wandb.Image(vsutils.make_grid(fake.cpu(), padding=2, normalize=True,
                                                                    nrow=int(np.sqrt(utils.dpgan_batch_size))),
                                                  caption=f"Fake Samples at {epoch + 1}-{i + 1}")})

        disc_epsilon = disc_privacy_engine.accountant.get_epsilon(delta=utils.dpgan_discriminator_delta)

        run.log({
            "Loss Discriminator": error_D / len(data_loader),
            "Loss Generator": error_G / len(data_loader),
            "D(x)": D_x / len(data_loader),
            "D(G(z1))": D_G_z1 / len(data_loader),
            "D(G(z2))": D_G_z2 / len(data_loader),
            "Discriminator Privacy Cost": disc_epsilon,
        })

        utils.print_red(
            f"Epoch:{epoch} Loss Discriminator: {error_D / len(data_loader)} Loss Generator: {error_G / len(data_loader)} Generator Epsilon: {0} Discriminator Epsilon: {0}")

        torch.save(gen_net.state_dict(), f'{utils.DPGAN_MODEL_PATH}/Generator_Epoch_{epoch}.pth')
        torch.save(disc_net.state_dict(), f'{utils.DPGAN_MODEL_PATH}/Discriminator_Epoch_{epoch}.pth')

    artifact = wandb.Artifact('model', type='model')

    torch.save(gen_net.state_dict(), f'{utils.DPGAN_MODEL_PATH}/Generator.pth')
    torch.save(disc_net.state_dict(), f'{utils.DPGAN_MODEL_PATH}/Discriminator.pth')

    artifact.add_file(f'{utils.DPGAN_MODEL_PATH}/Generator.pth')
    artifact.add_file(f'{utils.DPGAN_MODEL_PATH}/Discriminator.pth')
    run.log_artifact(artifact)


if __name__ == "__main__":
    import os

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import wandb
    import warnings

    warnings.simplefilter("ignore")
    wandb.login(key="c8b7ef31a46dca526003891b3b6dda9f2a6391cf")

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="2412-DP GAN Model",
        entity="2412",
        config={
            "generator_learning_rate": utils.dpgan_gen_lr,
            "discriminator_learning_rate": utils.dpgan_disc_lr,
            "architecture": "DCGAN",
            "dataset": "MNIST",
            "optimizer": "DP SGD",
            "loss function": "Binary Cross Entropy",
            "epochs": utils.attack_epochs,
        },
    )

    train(run)

    run.finish()