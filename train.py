import torch, os, sys

from skimage import io, transform #type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
# from cifar.models import * 

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


class ToTensor(object):

    def __call__(self, sample):
       
        image, grid = sample["image"], sample["grid"]

        image = image.transpose((2, 0, 1))
        return {
            "image_name": sample["image_name"],
            "image": torch.from_numpy(image) / 255.0,
            "grid": torch.from_numpy(grid),
        }

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

size = 180
# seed = np.random.randint(4/5 * 180)

gridpoints_dataset = GridPointsDataset(
    category="Crosswalk",
    array_dir="data/labels/_ndarrays",
    image_dir="data/images",
    train=True,
    # seed=seed,
    transform=transforms.Compose([
        ToTensor(),
        ColorChannelCorrection(),
        ]),
)

from sklearn.model_selection import train_test_split

range_train, range_test = train_test_split(range(len(gridpoints_dataset)), test_size=0.2)

train_dataset = torch.utils.data.Subset(gridpoints_dataset, range_train)
val_dataset = torch.utils.data.Subset(gridpoints_dataset, range_test)

# val_dataset = GridPointsDataset(
#     category="Crosswalk",
#     array_dir="data/labels/_ndarrays",
#     image_dir="data/images",
#     train=False,
#     seed=seed,
#     transform=transforms.Compose([
#         ToTensor(),
#         ColorChannelCorrection(),
#         ]),
# )

training_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)   
# show_batch(training_dataloader, train=True)

# print(len(gridpoints_dataset), len(val_dataset))
# print(seed)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30 * 30 * 3, 512), #120/4 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
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
lr = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# break image into 16 smaller images corresponding to given grid
def split_image(image):
    image_grid = [transforms.functional.crop(image, (i // 4) * 30, (i % 4) * 30, 30, 30) for i in range(16)]
    return image_grid

images_batch = split_image(gridpoints_dataset[0]["image"])
image_grid = utils.make_grid(images_batch, nrow = 4)
plt.imshow(image_grid.numpy().transpose((1, 2, 0)))
plt.show(block=True)


def train(dataloader, valdataloader, model, loss_fn, optimizer, epochs=100):

    size = len(dataloader.dataset)
    model.train()
    losses = []
    lossavg = []
    valloss = []
    valavg = []
    for epoch in range(epochs):
        
        for batch, sample in enumerate(dataloader):

            image, grid = sample["image"].to(device), sample["grid"].to(device)
            i = 0
            for image_square in split_image(image):
                pred = model(image_square)
                
                grid_val = grid[0][i].unsqueeze(0).unsqueeze(0)
                
                loss = loss_fn(pred, grid_val.type(torch.float32))  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                i += 1
            if batch % 10 == 0:
                os.system('cls')
                print("epoch: ", epoch)
                meanloss, meanvalloss, current = np.mean(losses[-50:]), np.mean(valloss[-50:]), (batch + 1) * len(image)

                lossavg.append(meanloss)
                
                print(f"loss: {loss:>f} [{current:>5d}/{size:>5d}]\naverage validation loss: {meanvalloss:>f}\naverage loss: {meanloss:>f}\nlr: {lr}")
        if epoch % 5 == 4:
            for batch, sample in enumerate(valdataloader):

                image, grid = sample["image"].to(device), sample["grid"].to(device)
                i = 0
                for image_square in split_image(image):

                    pred = model(image_square)
                    grid_val = grid[0][i].unsqueeze(0).unsqueeze(0)
                
                    loss = loss_fn(pred, grid_val.type(torch.float32)) 

                    valloss.append(loss.item())
                    
                    meanloss = np.mean(valloss[-50:])
                    valavg.append(meanloss)
                    i += 1

    
    plot_loss(lossavg, epochs, valavg, max(max(lossavg), max(valavg)))
    return lossavg, valavg 

def plot_loss(lossavg, epochs, valavg, maxloss):
    plt.figure()
    plt.ioff()


    plt.plot(lossavg, label='average loss')
    plt.plot(valavg, label='average val loss')

    plt.axis((0, epochs, 0, maxloss))
    # print(max(loss))
    plt.legend()
    plt.show(block=True)
            
# model = SimpleDLA(num_classes=1).to(device)
# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
    
# model.load_state_dict(torch.load("./cifar/checkpoint/ckpt.pth")['net'], strict=False)
train_set_testing = set([x['image_name'] for x in train_dataset])
val_set_testing = set([x['image_name'] for x in val_dataset])




assert train_set_testing.intersection(val_set_testing) == set(), "Train and validation sets are not disjoint"

lossavg, valavg  = train(training_dataloader, val_dataloader, model, loss_fn, optimizer, epochs=100)

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")