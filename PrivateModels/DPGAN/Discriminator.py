import torch
import torch.nn as nn

from utils import utils


class Discriminator(nn.Module):
    def __init__(self, num_gpus):
        super().__init__()
        self.num_gpus = num_gpus
        self.label_emb = nn.Embedding(10, 10)

        self.disc_net = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        if x.is_cuda and self.num_gpus > 1:
            out = nn.parallel.data_parallel(self.disc_net, x, range(self.num_gpus))
        else:
            out = self.disc_net(x)
        return out.squeeze()

# class Discriminator(nn.Module):
#     def __init__(self, num_channels, num_filters, num_gpus):
#         super(Discriminator, self).__init__()
#
#         self.num_gpus = num_gpus
#
#         self.disc_net = nn.Sequential(
#             # input size (1) * 28 * 28
#             nn.Conv2d(num_channels, num_filters, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # out size (64) * 14 * 14
#             nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_filters * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # out size (num_filters*2) * 7 * 7
#             nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_filters * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # out size (num_filters*2) * 3 * 3
#             nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_filters * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # out size (num_filters*4) * 1 * 1
#             nn.Conv2d(num_filters * 8, 1, 3, 2, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         if x.is_cuda and self.num_gpus > 1:
#             out = nn.parallel.data_parallel(self.disc_net, x, range(self.num_gpus))
#         else:
#             out = self.disc_net(x)
#         return out


if __name__ == "__main__":
    from DatasetGenerator.main import generate_dataset

    data_loader = generate_dataset(utils.DATA_DIR, batch_size=utils.attack_batch_size, img_size=utils.attack_img_size)

    x, y = next(iter(data_loader))
    # y = torch.full((128,), 1)

    # net = Discriminator(utils.num_channels, utils.num_attack_discriminator_filter, utils.num_gpus)
    net = Discriminator(utils.num_gpus)
    print(x.shape)
    y_preds = net(x, y)

    print(y_preds.shape, y.shape)
    # print(y)
