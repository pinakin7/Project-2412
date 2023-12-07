import numpy as np
import torch
from torch.nn.functional import  adaptive_avg_pool2d

from utils.statistics.InceptionV3 import InceptionV3


def calculate_activation_statistics(model,images, batch_size=128, dims=2048, cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        model = model.cuda()
        batch = images.cuda()
    else:
        batch = images
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    model = model.cpu()
    return mu, sigma

if __name__ == "__main__":
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.randn((128,1,32,32)).to('cuda' if torch.cuda.is_available() else 'cpu')
    print(calculate_activation_statistics(model,images, cuda=torch.cuda.is_available()))