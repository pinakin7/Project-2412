import numpy as np
import torch
import PIL.Image as Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import os


class DatasetFID(Dataset):
    def __init__(self, real_image_file, fake_image_file, is_dir=False):
        self.real_images = torch.tensor(np.load(real_image_file)).expand(-1, 3, -1, -1)
        image_array = np.load(fake_image_file) if not is_dir else self._get_arrays(fake_image_file)
        self.fake_images = torch.tensor(image_array).permute(0, 3, 1, 2).expand(-1, 3, -1, -1)
        # self.fake_images = torch.tensor(np.load(fake_image_file)).expand(-1,3,-1,-1)

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        return self.real_images[idx], self.fake_images[idx]

    def _get_arrays(self, root_folder):
        image_data = []

        # Get the list of subdirectories (labels) in the root folder
        subdirectories = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

        for label in subdirectories:
            label_path = os.path.join(root_folder, label)

            # Iterate through each image in the label directory
            for filename in os.listdir(label_path):
                if filename.endswith(('.npy')):  # Add more image extensions if needed
                    file_path = os.path.join(label_path, filename)

                    # Read the image using OpenCV
                    image = np.load(file_path)

                    # You may want to resize the images to a consistent size
                    # image = cv2.resize(image, (desired_width, desired_height))

                    # Append the image data and label to the lists
                    image_data.append(image)
        return np.expand_dims(np.array(image_data), axis=-1)

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def compute_frechet_inception_distance(dataset, batch_size=100, save_plot=False, save_log=False, data_origin="None"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid = fid.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    fids = []
    for data in tqdm(dataloader, desc=f"Computing Frechet Inception for {data_origin}"):
        real_images, fake_images = data
        real_images = interpolate(real_images)
        real_images = real_images.to(device)
        fake_images = interpolate(fake_images)
        fake_images = fake_images.to(device)

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        curr_fid = fid.compute()
        fids.append(curr_fid.item())
    return fids


if __name__ == "__main__":
    dataset = DatasetFID(real_image_file="output/original/images.npy",
                         fake_image_file="output/diffusion/images.npy", is_dir=False)

    fid = compute_frechet_inception_distance(dataset, data_origin="DPGAN", batch_size=100)
    print(fid)
