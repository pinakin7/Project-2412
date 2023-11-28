import torch
import torch.nn as nn

from utils import utils


class Generator(nn.Module):
    def __init__(self, noise_dims: int, num_filters: int, num_channels: int, num_gpus: int):
        super(Generator, self).__init__()
        self.num_gpus = num_gpus
        self.gen_net = nn.Sequential(
            # the noise input is going into the convolution
            nn.ConvTranspose2d(noise_dims, num_filters * 8, 4, 1, 0, bias=False),
            nn.GroupNorm(min(32, num_filters * 8), num_filters*8),
            nn.ReLU(inplace=True),

            # now out size is (num_filters*8)*4*4
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, num_filters * 4), num_filters*4),
            nn.ReLU(inplace=True),

            # now out size is (num_filters*4)*8*8
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, num_filters * 2), num_filters * 2),
            nn.ReLU(inplace=True),

            # now out size is (num_filters*2)*16*16
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, num_filters), num_filters),
            nn.ReLU(inplace=True),

            # now out size is (num_filters)*16*16
            nn.ConvTranspose2d(num_filters, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),

            # now out size is (num_channels)*32*32
        )

    def forward(self, x):
        if x.is_cuda and self.num_gpus > 1:
            out = nn.parallel.data_parallel(self.gen_net, x, range(self.num_gpus))
        else:
            out = self.gen_net(x)

        return out

if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    net = Generator(utils.dpgan_noise_dim, utils.num_dpgan_generator_filter, utils.num_channels,
                    utils.num_gpus).to(utils.device)
    net.apply(utils.init_params_dpgan_model)
    noise = torch.randn(1, utils.dpgan_noise_dim, 1, 1, device=utils.device)

    out = net(noise)

    print(out[0].shape)
    plt.imshow(np.transpose(out.cpu().detach().squeeze().numpy(), (1, 2, 0)))
    plt.show()