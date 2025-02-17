

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


class ToTensor_train(object):

    def __call__(self, sample):
       
        image, grid = sample["image"], sample["grid"]

        image = image.transpose((2, 0, 1))
        return {
            "image_name": sample["image_name"],
            "image": torch.from_numpy(image) / 255.0,
            "grid": torch.from_numpy(grid),
        }
    


class ToTensor_test(object):

    def __call__(self, sample):
    
        image = sample["image"]

        image = image.transpose((2, 0, 1))
        return {
            "image_name": sample["image_name"],
            "image": torch.from_numpy(image) / 255.0,
        }


class GridPointsDataset(Dataset):

    def __init__(self, category, array_dir, image_dir, transform=None):

        self.category = category  # Crosswalk, Chimney, Stair
        self.array_dir = array_dir

        array_file = os.path.join(array_dir, f"{category}.npy")
        self.grids = np.load(array_file)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return self.grids.shape[0]

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

def show_grid_batch(sample_batched, train):
    print(f"train: {train}")
    images_batch = sample_batched["image"]
    if train: grid_batch =  sample_batched["grid"]

    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    image_grid_border_size = 0

    image_grid = utils.make_grid(images_batch)
    plt.imshow(image_grid.numpy().transpose((1, 2, 0)))

    plt.xticks(np.arange(0, 120, 30))
    plt.yticks(np.arange(0, 120, 30))
    plt.grid()

    ax = plt.gca()

    if train:
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

def show_batch(dataloader, train):
    if __name__ == "__main__":
        for i_batch, sample_batched in enumerate(dataloader):
            if train: 
                print("train", i_batch, sample_batched["image"].size(), sample_batched["grid"].size())
            else: 
                print("test", i_batch, sample_batched["image"].size())

            if i_batch == 3 or sample_batched != False:
                plt.figure()
                show_grid_batch(sample_batched, train)
                plt.title(f"Train: {train}")
                plt.axis("off")
                plt.ioff()
                plt.show(block=True)
                break

gridpoints_dataset = GridPointsDataset(
    category="Crosswalk",
    array_dir="data/labels/_ndarrays",
    image_dir="data/images",
    transform=transforms.Compose([ToTensor_train()]),
)

image_dataset = ImageDataset(
    category="Crosswalk",
    image_dir="data/images",
    transform=transforms.Compose([ToTensor_test()]),
)

training_dataloader = DataLoader(gridpoints_dataset, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=0)

   
show_batch(training_dataloader, train=True)
show_batch(test_dataloader, train=False)

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
            nn.Linear(120 * 120 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# model = NeuralNetwork().to(device)
# print(model)

# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lr = 1e-4

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def train(dataloader, model, loss_fn, optimizer, epochs=100):
    # scheduled_lr = 0
    # ideal start: 4e-3

    size = len(dataloader.dataset)
    model.train()
    losses = []
    lossavg= []
    epochavg = []
    for epoch in range(epochs):
        
        # if epoch % 10 == 0: scheduled_lr += 1e-7
        # ideal end: 2e-2
        for batch, sample in enumerate(dataloader):

            image, grid = sample["image"].to(device), sample["grid"].to(device)

            pred = model(image)
            loss = loss_fn(pred, grid.type(torch.float32))  

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if batch % 10 == 0:
                os.system('cls')
                print("epoch: ", epoch)
                loss, current = np.mean(losses[-50:]), (batch + 1) * len(image)
                lossavg.append(loss)
                epochavg.append(epoch)
                print(f"loss: {loss:>f} [{current:>5d}/{size:>5d}]\nlr: {lr}")
    
    plot_loss(losses, epochs, lossavg, epochavg)

def plot_loss(loss, epochs, lossavg, epochavg):
    plt.figure()
    plt.ioff()
    plt.plot(loss)
    plt.plot(epochavg, lossavg)
    plt.axis((0, epochs, 0, max(loss)))
    # print(max(loss))
    plt.show(block=True)
            

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            image, grid = sample["image"].to(device), sample["grid"].to(device)

            pred = model(image)
            print(torch.round(pred).type(torch.IntTensor))
            test_loss += loss_fn(torch.round(pred).type(torch.IntTensor), torch.round(grid)).item()
            print(pred, grid)
            correct += (pred.argmax(1) == grid).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )



# train(training_dataloader, model, loss_fn, optimizer, epochs=250)

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

test(training_dataloader, model, loss_fn)

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
