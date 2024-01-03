import torch

from torchvision.transforms import Compose, ToTensor

import datasets


class PoloClubDiffusionDBDataSet(torch.utils.data.Dataset):
    """A simple wrapper around the PoloClub Diffusion DB data set."""

    def __init__(self, transform=None):
        self._hf_dataset = datasets.load_dataset("poloclub/diffusiondb", "large_random_1k")["train"]

        if transform:
            self._transform = transform
        else:
            self._transform = Compose([
                ToTensor(),
            ])

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, idx):
        item = self._hf_dataset[idx]
        image = self._transform(item["image"])
        return image, item["prompt"]
