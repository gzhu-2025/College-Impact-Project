import torch, os, sys

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils

import warnings

warnings.filterwarnings("ignore")

plt.ion()

# if (sys.argv[1] != "Chimney"
#     and sys.argv[1] != "Crosswalk"
#     and sys.argv[1] != "Stair"
#     ):
#     raise Exception("Labels not found")

# label_path = f"data/labels/_ndarrays/{sys.argv[1]}.npy"

# n = 37

# grid = np.load(label_path)[n - 1]

# # print(f"grid {grid[0]}: ")
# n = grid[0]

# grid = grid[1:]

# image_name = f"data/images/{sys.argv[1]}/{'Cross' if sys.argv[1] == 'Crosswalk' else sys.argv[1]} ({n}).png"

# # for i in range(4):
#     for j in range(4):
#         print(grid[i * 4 + j], end=" ")
#     print()


def show_image(image, grid):
    fig = plt.gcf()
    ax = plt.gca()

    # plt.title(image_name)
    plt.imshow(image)

    plt.xticks(np.arange(0, 120, 30))
    plt.yticks(np.arange(0, 120, 30))
    plt.grid()

    for i in range(16):
        if grid[i]:
            row = int(i / 4)
            col = i % 4

            rect = patches.Rectangle(
                (col * 30, row * 30),
                30,
                30,
                linewidth=1,
                edgecolor="#1a73e8",
                facecolor="#1a73e8",
                alpha=0.3,
            )
            ax.add_patch(rect)


class ToTensor(object):

    def __call__(self, sample):
        image, grid = sample["image"], sample["grid"]

        image = image.transpose((2, 0, 1))
        return {
            "image_name": sample["image_name"],
            "image": torch.from_numpy(image) / 255.0,
            "grid": torch.from_numpy(grid),
        }


# plt.figure()
# # print(image_name)
# show_image(image_name, io.imread(image_name))

# plt.show(block=True)


# gridpoints_dataset = GridPointsDataset(category='Crosswalk', array_dir='data/labels/_ndarrays', image_dir='data/images', transform=transforms.Compose([
#                                                                                                                                         ToTensor()
#                                                                                                                                     ]))
# dataloader = DataLoader(gridpoints_dataset, batch_size=4, shuffle=True, num_workers=0)
class GridPointsDataset(Dataset):

    def __init__(self, category, array_dir, image_dir, transform=None):

        self.category = category  # Crosswalk, Chimney, Stair
        self.array_dir = array_dir

        array_file = os.path.join(array_dir, f"{category}.npy")
        self.grids = np.load(array_file)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return gridpoints_dataset.grids.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

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


# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# gridpoints_dataset = GridPointsDataset(category='Crosswalk', array_dir='data/labels/_ndarrays', image_dir='data/images', transform=None)
# fig = plt.figure()

# for i, sample in enumerate(gridpoints_dataset):
#     print(i, sample['image'].shape, sample['grid'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title(f'Sample #{i}, {sample['image_name']}')
#     ax.axis('off')
#     show_image(**sample)

#     if i==3:
#         plt.show(block=True)
#         break
#

# gridpoints_dataset = GridPointsDataset(category='Crosswalk', array_dir='data/labels/_ndarrays', image_dir='data/images', transform=transforms.Compose([
#                                                                                                                                         ToTensor()
#                                                                                                                                     ]))
# for i, sample in enumerate(gridpoints_dataset):
#     print(i, sample['image'].size(), sample['grid'].size())

#     if i==3:
#         break


def show_grid_batch(sample_batched):
    images_batch, grid_batch = sample_batched["image"], sample_batched["grid"]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    image_grid_border_size = 0

    image_grid = utils.make_grid(images_batch)
    plt.imshow(image_grid.numpy().transpose((1, 2, 0)))

    plt.xticks(np.arange(0, 120, 30))
    plt.yticks(np.arange(0, 120, 30))
    plt.grid()

    ax = plt.gca()

    for i in range(batch_size):
        for j in range(16):

            if grid_batch[i][j].item():
                row = int(j / 4)
                col = j % 4

                rect = patches.Rectangle(
                    (col * 30 + 123 * i, row * 30),
                    30,
                    30,
                    linewidth=1,
                    edgecolor="#1a73e8",
                    facecolor="#1a73e8",
                    alpha=0.3,
                )
                ax.add_patch(rect)

        plt.title("Batch from dataloader")


gridpoints_dataset = GridPointsDataset(
    category="Crosswalk",
    array_dir="data/labels/_ndarrays",
    image_dir="data/images",
    transform=transforms.Compose([ToTensor()]),
)
dataloader = DataLoader(gridpoints_dataset, batch_size=4, shuffle=True, num_workers=0)

if __name__ == "__main__":
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched["image"].size(), sample_batched["grid"].size())

        if i_batch == 3:
            plt.figure()
            show_grid_batch(sample_batched)
            plt.axis("off")
            plt.ioff()
            plt.show(block=True)
            break

# batch_size = 64

# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(), sample_batched['image'].dtype,
#             sample_batched['grid'].size(), sample_batched['grid'].dtype)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(120 * 120 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
# print(model)

# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


def train(dataloader, model, loss_fn, optimizer, epochs=100):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        for batch, sample in enumerate(dataloader):
            image, grid = sample["image"].to(device), sample["grid"].to(device)

            pred = model(image)
            loss = loss_fn(pred, grid.type(torch.float32))  #

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(image)
                print(f"loss: {loss:>f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            image, grid = sample["image"].to(device), sample["grid"].to(device)

            pred = model(image)
            test_loss += loss_fn(pred, grid).item()
            print(pred, grid)
            correct += (pred.argmax(1) == grid).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t + 1}\n----------------------------------------")
#     train(dataloader, model, loss_fn, optimizer)
#     test(dataloader, model, loss_fn)
# print("Done!")

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))

# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
