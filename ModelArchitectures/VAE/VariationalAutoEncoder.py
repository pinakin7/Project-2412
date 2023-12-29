import torch.nn as nn
import torch

from ModelArchitectures.VAE.Decoder import Decoder
from ModelArchitectures.VAE.Encoder import Encoder


def reparameterize(mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    epsilon = torch.rand_like(sigma)

    return mu + epsilon*sigma


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditional=False, num_classes=0):
        super(VariationalAutoencoder, self).__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, num_classes)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, num_classes)

    def forward(self, x, c=None):
        if x.dim() > 2:
            x = x.view(-1, 28*28)
        means, log_var = self.encoder(x, c)
        z = reparameterize(means, log_var)
        reconstruct_x = self.decoder(z, c)

        return reconstruct_x, means, log_var, z

    def inference(self, z, c=None):
        reconstruct_x = self.decoder(z, c)
        return reconstruct_x

