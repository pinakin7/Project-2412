import torch.nn as nn

import utils.utils
import torch

# class Generator(nn.Module):
#     def __init__(self, noise_dims: int, num_filters: int, num_channels: int, num_gpus: int):
#         super(Generator, self).__init__()
#
#         self.num_gpus = num_gpus
#
#         self.gen_net = nn.Sequential(
#
#             # the noise input is going into the convolution
#             nn.ConvTranspose2d(noise_dims, num_filters * 8, 3, 2, 0, bias=False),
#             nn.BatchNorm2d(num_filters * 8),
#             nn.ReLU(inplace=True),
#
#             # now out size is (num_filters*8)*4*4
#             nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 3, 2, 2, bias=False),
#             nn.BatchNorm2d(num_filters * 4),
#             nn.ReLU(inplace=True),
#
#             # now out size is (num_filters*4)*8*8
#             nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(num_filters * 2),
#             nn.ReLU(inplace=True),
#
#             # now out size is (num_filters*2)*16*16
#             nn.ConvTranspose2d(num_filters * 2, num_filters, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(inplace=True),
#
#             nn.ConvTranspose2d(num_filters, num_channels, 3, 2, 2, bias=False),
#             nn.BatchNorm2d(num_channels),
#             nn.ReLU(inplace=True),
#
#             # now out size is (num_filters)*16*16
#             nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 2, bias=False),
#             nn.Tanh(),
#
#             # now out size is (num_channels)*32*32
#         )
#
#     def forward(self, x):
#         if x.is_cuda and self.num_gpus > 1:
#             out = nn.parallel.data_parallel(self.gen_net, x, range(self.num_gpus))
#         else:
#             out = self.gen_net(x)
#
#         return out


class Generator(nn.Module):
    def __init__(self, num_gpus):
        super().__init__()
        self.num_gpus = num_gpus
        self.label_emb = nn.Embedding(10, 10)

        self.gen_net = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        # print(c.shape)
        x = torch.cat([z, c], 1)
        batch_size = x.size(0)
        if x.is_cuda and self.num_gpus > 1:
            out = nn.parallel.data_parallel(self.gen_net, x, range(self.num_gpus))
        else:
            out = self.gen_net(x)
        # out = self.gen_net(x)
        return out.view(batch_size, 28, 28)


if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    net = Generator(1).cuda()
    noise = torch.autograd.Variable(torch.randn(16, 100)).cuda()
    labels = torch.autograd.Variable(torch.LongTensor(np.random.randint(0, 10, 16))).cuda()
    out = net(noise, labels)

    print(out.shape)
    # plt.imshow(np.transpose(out.cpu().detach().squeeze().numpy(), (1, 2, 0)))
    # plt.show()
