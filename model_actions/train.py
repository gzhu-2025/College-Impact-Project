import torch, os, sys

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from utils import *
from models import *
from transforms import * 
from datasets import *
from .test import *

import warnings

warnings.filterwarnings("ignore")

plt.ion()


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

# from sklearn.model_selection import train_test_split

# range_train, range_test = train_test_split(range(len(gridpoints_dataset)), test_size=0.2)

# train_dataset = torch.utils.data.Subset(gridpoints_dataset, range_train)
# val_dataset = torch.utils.data.Subset(gridpoints_dataset, range_test)

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

training_dataloader = DataLoader(gridpoints_dataset, batch_size=1, shuffle=True, num_workers=0)
# val_dataloader = DataLoader(gridpoints_dataset, batch_size=1, shuffle=True, num_workers=0)   
show_batch(training_dataloader, train=True)

# print(len(gridpoints_dataset), len(val_dataset))
# print(seed)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# print(f"Using {device} device")

model = NeuralNetwork().to(device)
# print(model)

# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lr = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

        val = test(valdataloader, model, loss_fn)
        valloss.append(val.item())
        valavg.append(np.mean(valloss[-50:]))

    
    plot_loss(lossavg, epochs, valavg, max(max(lossavg), max(valavg)))
    return lossavg, valavg 

       
# model = SimpleDLA(num_classes=1).to(device)
# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
    
# model.load_state_dict(torch.load("./cifar/checkpoint/ckpt.pth")['net'], strict=False)
# train_set_testing = set([x['image_name'] for x in train_dataset])
# val_set_testing = set([x['image_name'] for x in val_dataset])




# assert train_set_testing.intersection(val_set_testing) == set(), "Train and validation sets are not disjoint"

# lossavg, valavg  = train(training_dataloader, val_dataloader, model, loss_fn, optimizer, epochs=100)

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")