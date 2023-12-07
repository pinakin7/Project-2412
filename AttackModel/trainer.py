import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vsutils
from tqdm import tqdm

from AttackModel.Discriminator import Discriminator
from AttackModel.Generator import Generator
from DatasetGenerator.main import generate_dataset
from utils import utils


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = torch.autograd.Variable(torch.randn(batch_size, utils.dpgan_noise_dim)).cuda()
    fake_labels = torch.autograd.Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, torch.autograd.Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.autograd.Variable(torch.ones(batch_size)).cuda())

    # train with fake images
    z = torch.autograd.Variable(torch.randn(batch_size, utils.dpgan_noise_dim)).cuda()
    fake_labels = torch.autograd.Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.autograd.Variable(torch.zeros(batch_size)).cuda())

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train(run):
    data_loader = generate_dataset(utils.DATA_DIR, batch_size=utils.attack_batch_size, img_size=utils.attack_img_size)

    gen_net = Generator(utils.num_gpus).to(utils.device)
    # gen_net.apply(utils.init_params_attack_model)

    disc_net = Discriminator(utils.num_gpus).to(utils.device)
    # disc_net.apply(utils.init_params_attack_model)

    criterion = nn.BCELoss()

    optimizer_gen = optim.Adam(gen_net.parameters(), lr=utils.attack_gen_lr, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(disc_net.parameters(), lr=utils.attack_disc_lr, betas=(0.5, 0.999))

    fixed_noise = torch.autograd.Variable(torch.randn(utils.attack_batch_size, 100)).cuda()

    run.watch(gen_net, log='all', log_freq=100)
    run.watch(disc_net, log='all', log_freq=100)

    for epoch in range(utils.attack_epochs):
        error_d, error_G = 0, 0
        for i, (data, labels) in tqdm(enumerate(data_loader, 0), desc=f"Epoch {epoch}/{utils.attack_epochs}:",
                                      total=len(data_loader)):
            gen_net.train()
            disc_net.train()
            ########################################################
            # (Step 1): Updating the Discriminator model:
            # # maximize log(D(x)) + log(1 - D(G(Z)))
            ########################################################
            batch_size = data.size(0)
            # training with the real data
            data = torch.autograd.Variable(data).cuda()
            labels = torch.autograd.Variable(labels).cuda()

            for _ in range(utils.attack_num_critic):
                d_loss = discriminator_train_step(batch_size=batch_size, discriminator=disc_net, d_optimizer=optimizer_disc,generator=gen_net, real_images=data, labels=labels, criterion=criterion)
                error_d+=d_loss
                run.log({"Discriminator Loss": d_loss})

            ########################################################
            # (Step 2): Updating the Generator model:
            # # maximize log(D(G(Z)))
            ########################################################
            g_loss = generator_train_step(batch_size=batch_size, discriminator=disc_net, generator=gen_net, criterion=criterion, g_optimizer=optimizer_gen)
            error_G+=g_loss
            run.log({"Generator Loss": g_loss})

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    fake = gen_net(fixed_noise, labels).unsqueeze(1).detach()
                    run.log({"Generated Images": wandb.Image(
                        vsutils.make_grid(0.3081*fake.cpu()+0.1307, padding=2, normalize=True, nrow=10),
                                                  caption=f"Fake Samples at {epoch}-{i + 1}"),
                             "Original Images": wandb.Image(
                                 vsutils.make_grid(data.cpu(), padding=2, normalize=True, nrow=10),
                                 caption=f"Original Samples at {epoch}-{i + 1}")
                             })

        utils.print_red(
            f"Epoch:{epoch} Loss Discriminator: {error_d / len(data_loader)} Loss Generator: {error_d / len(data_loader)}")
        run.log({"Epoch Generator Loss": error_G/len(data_loader), "Epoch Discrimnator Loss": error_d/(len(data_loader)*utils.attack_num_critic)})

        torch.save(gen_net.state_dict(), f'{utils.ATTACK_MODEL_PATH}/Generator_Epoch_{epoch}.pth')
        torch.save(disc_net.state_dict(), f'{utils.ATTACK_MODEL_PATH}/Discriminator_Epoch_{epoch}.pth')

    artifact = wandb.Artifact('model', type='model')

    torch.save(gen_net.state_dict(), f'{utils.ATTACK_MODEL_PATH}/Generator.pth')
    torch.save(disc_net.state_dict(), f'{utils.ATTACK_MODEL_PATH}/Discriminator.pth')

    artifact.add_file(f'{utils.ATTACK_MODEL_PATH}/Generator.pth')
    artifact.add_file(f'{utils.ATTACK_MODEL_PATH}/Discriminator.pth')
    run.log_artifact(artifact)


if __name__ == "__main__":
    import wandb

    wandb.login(key="c8b7ef31a46dca526003891b3b6dda9f2a6391cf")

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="2412-Attack Model",
        entity="2412",
        config={
            "generator_learning_rate": utils.attack_gen_lr,
            "discriminator_learning_rate": utils.attack_disc_lr,
            "architecture": "DCGAN",
            "dataset": "CIFAR-10",
            "optimizer": "Adam",
            "loss function": "Binary Cross Entropy",
            "epochs": utils.attack_epochs,
        },
    )

    train(run)

    run.finish()
