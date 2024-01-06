import torch

from torchvision.transforms import Compose, ToTensor

import datasets


class HuggingFaceImageDataSet(torch.utils.data.Dataset):
    """A simple torch Dataset wrapper around a Hugging Face image-based data set."""

    def __init__(self, hf_dataset: datasets.DatasetDict, transform=None):
        self._hf_dataset = hf_dataset

        if transform:
            self._transform = transform
        else:
            self._transform = Compose(
                [
                    ToTensor(),
                ]
            )

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, idx):
        item = self._hf_dataset[idx]
        image = self._transform(item["image"])
        return image, item["label"]
