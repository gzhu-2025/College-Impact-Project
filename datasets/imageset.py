import torch
from torch.utils.data import Dataset

import os

from skimage import io

class ImageDataset(Dataset):

    def __init__(self, category, image_dir, transform=None):

        self.category = category  # Crosswalk, Chimney, Stair

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        image_path = os.path.normpath(
            os.path.join(
                self.image_dir,
                self.category,
            )
        )

        image_name = os.listdir(image_path)[index]

        image_name = os.path.normpath(
            os.path.join(
                image_path, 
                image_name,
            )
        )

        image = io.imread(image_name)

        sample = {"image_name": image_name, "image": image}

        if self.transform:
            sample = self.transform(sample)

        return sample
