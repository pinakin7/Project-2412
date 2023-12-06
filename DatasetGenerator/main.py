import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

random.seed(100)
torch.manual_seed(100)


def generate_dataset(data_dir: str = "./data", img_size: int = 64, batch_size: int = 128,
                     num_workers: int = 2, train: bool = True) -> data.dataloader.DataLoader:
    dataset = datasets.MNIST(root=data_dir, download=True, transform=transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307),
                             (0.3081)),
    ]), train=train)

    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


if __name__ == "__main__":
    x = generate_dataset(batch_size=16)

    print(type(x))

    from utils import utils

    img, labels = next(iter(x))

    print(img.shape)

    utils.plot_images_grid(16, 64, img, labels, utils.label_class)
