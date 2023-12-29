import torch
import torch.nn as nn

import utils.utils


class Critic(nn.Module):
    def __init__(self, num_classes, critic_size, critic_hidden_size):
        super(Critic, self).__init__()
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_classes, critic_size * 4),
        )
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1, critic_size // 4, 3, 2),
            nn.InstanceNorm2d(critic_size // 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(critic_size // 4, critic_size // 2, 3, 2),
            nn.InstanceNorm2d(critic_size // 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(critic_size // 2, critic_size, 3, 2),
            nn.InstanceNorm2d(critic_size, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.Critic_net = nn.Sequential(
            nn.Linear(critic_size * 8, critic_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(critic_hidden_size, 1),
        )

    def forward(self, image, condition):
        vec_condition = self.condition_embedding(condition)
        cnn_features = self.cnn_net(image)
        combined = torch.cat([cnn_features, vec_condition], dim=1)
        return self.Critic_net(combined)

if __name__ == "__main__":
    from DatasetGenerator.main import generate_dataset
    from utils.utils import GANHyperparameter

    loader = generate_dataset(utils.utils.DATA_DIR)
    x, _ = next(iter(loader))
    y = torch.full((128,), 1)

    hp = GANHyperparameter()

    net = Critic(hp.num_classes, hp.critic_size, hp.critic_hidden_size)

    y_preds = net(x)

    print(y_preds.view(-1).shape, y.shape)
