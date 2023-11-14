import torch

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from utils import utils
from Discriminator import Discriminator
from Generator import Generator
from DatasetGenerator.main import generate_dataset


def train(wandb):
    data_loader = generate_dataset(utils.DATA_DIR, batch_size=utils.attack_batch_size)

    gen_net = Generator(utils.attack_noise_dim, utils.num_attack_generator_filter, utils.num_channels,
                        utils.num_gpus).to(utils.device)
    gen_net.apply(utils.init_params_attack_model)

    disc_net = Discriminator(utils.num_channels, utils.num_attack_discriminator_filter, utils.num_gpus).to(utils.device)
    disc_net.apply(utils.init_params_attack_model)

    criterion = nn.BCELoss()

    optimizer_gen = optim.Adam(gen_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(disc_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    fixed_noise = torch.randn(utils.attack_batch_size, utils.attack_noise_dim, 1, 1, device=utils.device)

    for epoch in range(utils.attack_epochs):
        error_D, error_G, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0
        for i, data in tqdm(enumerate(data_loader, 0), desc=f"Epoch {epoch}/{utils.attack_epochs}:",
                            total=len(data_loader)):
            ########################################################
            # (Step 1): Updating the Discriminator model:
            # # maximize log(D(x)) + log(1 - D(G(Z)))
            ########################################################
            data = data[0]
            batch_size = data.size(0)
            # training with the real data
            disc_net.zero_grad()
            data = data.to(utils.device)
            label = torch.full((batch_size,), real_label, device=utils.device, dtype=torch.float32)

            output = disc_net(data)
            err_disc_real = criterion(output, label)
            err_disc_real.backward()
            D_x += output.mean().item()

            # training with fake data
            noise = torch.randn(batch_size, utils.attack_noise_dim, 1, 1, device=utils.device)
            fake = gen_net(noise)
            label.fill_(fake_label)
            output = disc_net(fake.detach())
            err_disc_fake = criterion(output, label)
            err_disc_fake.backward()
            D_G_z1 += output.mean().item()
            error_D += err_disc_fake + err_disc_real
            optimizer_disc.step()

            ########################################################
            # (Step 2): Updating the Generator model:
            # # maximize log(D(G(Z)))
            ########################################################
            gen_net.zero_grad()
            label.fill_(real_label)
            output = disc_net(fake)
            err_gen = criterion(output, label)
            err_gen.backward()
            D_G_z2 += output.mean().item()
            error_G += err_gen
            optimizer_gen.step()

            if i % 100 == 0:
                wandb.log({"image": wandb.Image(data[0].squeeze().cpu().numpy().transpose(1, 2, 0),
                                                caption=f"Real Sample at {epoch}-{i}")})
                fake = gen_net(fixed_noise).detach()
                wandb.log({"image": wandb.Image(fake[0].squeeze().cpu().numpy().transpose(1, 2, 0),
                                                caption=f"Fake Sample at {epoch}-{i}")})

        utils.print_red(
            f"Epoch:{epoch} Loss Discriminator: {error_D / len(data_loader)} Loss Generator: {error_G / len(data_loader)} D(x): {D_x / len(data_loader)} D(G(z1)): {D_G_z1 / len(data_loader)} D(G(z2)): {D_G_z2 / len(data_loader)}")

        wandb.log({
            "Loss Discriminator": error_D / len(data_loader),
            "Loss Generator": error_G / len(data_loader),
            "D(x)": D_x / len(data_loader),
            "D(G(z1))": D_G_z1 / len(data_loader),
            "D(G(z2))": D_G_z2 / len(data_loader),
        })

        torch.save(gen_net.state_dict(), f'{utils.ATTACK_MODEL_PATH}/weights/Generator_Epoch_{epoch}.pth')
        torch.save(disc_net.state_dict(), f'{utils.ATTACK_MODEL_PATH}/weights/Discriminator_Epoch_{epoch}.pth')

    wandb.log({"generator_model_summary": wandb.Histogram(gen_net.summary()),
               "discriminator_model_summary": wandb.Histogram(disc_net.summary())})


if __name__ == "__main__":
    import wandb

    wandb.init(
        # set the wandb project where this run will be logged
        project="2412-Attack Model",
        config={
            "generator_learning_rate": 0.0002,
            "discriminator_learning_rate": 0.0002,
            "architecture": "DCGAN",
            "dataset": "CIFAR-10",
            "optimizer": "Adam",
            "loss function": "Binary Cross Entropy",
            "epochs": 30,
        },
    )

    train(wandb)

    wandb.finish()
