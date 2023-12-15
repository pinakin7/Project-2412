import numpy as np
import torch
from torch.utils.data import Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


class DatasetFID(Dataset):
    def __init__(self, real_image_file, fake_image_file):
        self.real_images = torch.tensor(np.load(real_image_file)).expand(-1,3,-1,-1)
        # self.fake_images = torch.tensor(np.load(fake_image_file)).permute(0,3,1,2).expand(-1,3,-1,-1)
        self.fake_images = torch.tensor(np.load(fake_image_file)).expand(-1,3,-1,-1)

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        return self.real_images[idx], self.fake_images[idx]


def compute_frechet_inception_distance(dataset, batch_size=100, save_plot=False, save_log=False, data_origin="None"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid = fid.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    fids = []
    for data in tqdm(dataloader, desc=f"Computing Frechet Inception for {data_origin}"):
        real_images, fake_images = data
        real_images = real_images.to(device)
        fake_images = fake_images.to(device)

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        curr_fid = fid.compute()
        fids.append(curr_fid.item())
    return fids

if __name__ == "__main__":
    dataset = DatasetFID(real_image_file="output/original/images.npy", fake_image_file="output/original/images.npy")

    fid = compute_frechet_inception_distance(dataset, data_origin="WGAN", batch_size=100)
    print(fid)
