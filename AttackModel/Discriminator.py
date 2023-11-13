import torch
import torch.nn as nn

import utils.utils


class Discriminator(nn.Module):
    def __init__(self, num_channels, num_filters, num_gpus):
        super(Discriminator, self).__init__()

        self.num_gpus = num_gpus

        self.disc_net = nn.Sequential(
            # input size (num_channels) * 64 * 64
            nn.Conv2d(num_channels, num_filters, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # out size (num_filters) * 32 * 32
            nn.Conv2d(num_filters, num_filters*2, 4,2,1,bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU(0.2, inplace=True),

            # out size (num_filters*2) * 16 * 16
            nn.Conv2d(num_filters*2, num_filters*4,4,2,1,bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU(0.2, inplace=True),

            # out size (num_filters*4) * 8 * 8
            nn.Conv2d(num_filters*4, num_filters*8,4,2,1,bias=False),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU(0.2, inplace=True),

            # out size (num_filters*8) * 4 * 4
            nn.Conv2d(num_filters*8, 1,4,2,0,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.is_cuda and self.num_gpus>1:
            out = nn.parallel.data_parallel(self.disc_net, x, range(self.num_gpus))
        else:
            out = self.disc_net(x)
        return out.view(-1,1).squeeze()


if __name__ == "__main__":
    from DatasetGenerator.main import generate_dataset

    loader = generate_dataset(utils.utils.DATA_DIR)

    x,_ = next(iter(loader))
    y = torch.full((128,), 1)

    net = Discriminator(utils.utils.num_channels,utils.utils.num_attack_discriminator_filter,utils.utils.num_gpus)

    y_preds = net(x)

    print(y_preds,y)