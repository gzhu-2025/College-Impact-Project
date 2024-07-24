import torch

class ToTensor(object):

    def __call__(self, sample):
       
        image, grid = sample["image"], sample["grid"]

        image = image.transpose((2, 0, 1))
        return {
            "image_name": sample["image_name"],
            "image": torch.from_numpy(image) / 255.0,
            "grid": torch.from_numpy(grid),
        }
