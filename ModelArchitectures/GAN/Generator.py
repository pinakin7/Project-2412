import torch.nn as nn

import utils.utils


class Generator(nn.Module):
    def __init__(self, num_classes, generator_size, latent_size):
        super(Generator, self).__init__()
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_size, generator_size // 2),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_classes, generator_size // 2),
        )
        self.tcnn = nn.Sequential(
            nn.ConvTranspose2d(generator_size, generator_size, 4, 1, 0),
            nn.BatchNorm2d(generator_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(generator_size, generator_size // 2, 3, 2, 1),
            nn.BatchNorm2d(generator_size // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(generator_size // 2, generator_size // 4, 4, 2, 1),
            nn.BatchNorm2d(generator_size // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(generator_size // 4, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, latent, condition):
        vec_latent = self.latent_embedding(latent)
        vec_class = self.condition_embedding(condition)
        combined = torch.cat([vec_latent, vec_class], dim=1).reshape(-1, hp.generator_size, 1, 1)
        return self.tcnn(combined)


if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.utils import GANHyperparameter
    hp = GANHyperparameter()
    net = Generator(hp.num_classes, hp.generator_size, hp.latent_size)

    noise = torch.randn(1, utils.utils.attack_noise_dim, 1, 1, device=utils.utils.device)

    out = net(noise)

    print(out.shape)
    plt.imshow(np.transpose(out.cpu().detach().squeeze().numpy(), (1, 2, 0)))
    plt.show()
