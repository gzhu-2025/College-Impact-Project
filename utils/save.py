import torch, os
from skimage import io  # type: ignore

image_directories = os.listdir("data/images")

X = []
y = []
current_int = 0
for dir in image_directories:
    current_directory = f"data/images/{dir}"
    current_category = dir

    current_images = []
    for im in os.listdir(current_directory):
        im = io.imread(f"{current_directory}/{im}")
        if im.shape[0] == 120 and im.shape[1] == 120:
            current_images.append(torch.tensor(im[:, :, :3]))  # type: ignore
    X.append(torch.stack(current_images))
    y.extend([current_int] * X[-1].shape[0])
    current_int += 1
X = torch.cat(X, dim=0)
y = torch.tensor(y)

torch.save(X, 'X.pt')
torch.save(y, 'y.pt')