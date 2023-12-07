import torch.nn as nn

from NonPrivateModels.VAE.Decoder import Decoder
from NonPrivateModels.VAE.VariationalEncoder import VariationalEncoder


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)