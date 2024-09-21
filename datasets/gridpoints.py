import torch
from torch.utils.data import Dataset

import numpy as np
import os

from skimage import io #type: ignore

class GridPointsDataset(Dataset):

    def __init__(self, category, array_dir, image_dir, train, transform=None):
        self.train = train
        # self.seed = seed

        self.category = category  # Crosswalk, Chimney, Stair
        self.array_dir = array_dir

        array_file = os.path.join(array_dir, f"{category}.npy")
        self.grids = np.load(array_file)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # return int(self.grids.shape[0] * 4/5) \
        #     if self.train else int(self.grids.shape[0] /5) 
        return self.grids.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        # if self.train:
        #     if index > seed:
        #         num = int(index + self.__len__() / 4)
        #     else:
        #         num = index
        # else:
        #     num = index + self.seed

        image_name = os.path.normpath(
            os.path.join(
                self.image_dir,
                self.category,
                f"{self.category} ({self.grids[index][0]}).png",
            )
        )

        image = io.imread(image_name)

        grid = self.grids[index][1:]

        sample = {"image_name": image_name, "image": image, "grid": grid}

        if self.transform:
            sample = self.transform(sample)

        return sample
