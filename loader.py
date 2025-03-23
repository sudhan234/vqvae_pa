from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

dataset = load_dataset("bitmind/ffhq-256", cache_dir="./dataset", split='train')

class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]  # The dataset provides PIL images
        if self.transform:
            image = self.transform(image)  # Apply transformations
        return image


ffhq_dataset = FFHQDataset(dataset, transform=transform)
dataloader = DataLoader(ffhq_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)