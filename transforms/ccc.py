
# removes the 4th RGBa channel for models which use only 3 color channels
class ColorChannelCorrection(object):
    def __call__(self, sample):
        image, grid = sample["image"], sample["grid"]

        
        # zeros = torch.zeros(3, 120, 120)
        
        # zeros = zeros + image[:3]
        
        assert image[:3].shape == (3, 120, 120) , "Image shape is not 3, 120, 120"

        return {
            "image_name": sample["image_name"],
            "image": image[:3],
            "grid": grid,
        }

