import torch, os

import numpy as np
import matplotlib.pyplot as plt

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


def train(dataloader, valdataloader, model, loss_fn, optimizer, epochs=100):

    size = len(dataloader.dataset)
    model.train()
    losses = []
    lossavg = []
    valloss = []
    valavg = []
    for epoch in range(epochs):

        total = 0
        correct = 0
        for batch, sample in enumerate(dataloader):

            image, grid = sample["image"].to(device), sample["grid"].to(device)
            i = 0
            for image_square in split_image(
                image
            ):  # new_split_image: return tensor of shape 16 X 3 X 30 X 30
                grid_val = grid[0][i].unsqueeze(0).unsqueeze(0)

                optimizer.zero_grad()
                pred = model(image_square)
                loss = loss_fn(pred, grid_val.type(torch.float32))
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                _, predicted = pred.max(1)
                total += 1
                correct += predicted.eq(grid_val).sum().item()

                i += 1
            if batch % 10 == 0:
                os.system("cls")
                print("epoch: ", epoch)
                meanloss, meanvalloss, current = (
                    np.mean(losses[-50:]),
                    np.mean(valloss[-50:]),
                    (batch + 1) * len(image),
                )

                lossavg.append(meanloss)

                print(
                    f"loss: {loss:>f} [{current:>5d}/{size:>5d}]\naverage loss: {meanloss:>f}\naverage validation loss: {meanvalloss:>f}\nAccuracy: {correct/total * 100.}% ({correct}/{total})"
                )

        val, _ = test(valdataloader, model, loss_fn)
        model.train()
        valloss.append(val.item())
        valavg.append(np.mean(valloss[-50:]))

    plot_loss(
        model._get_name(), lossavg, epochs, valavg, max(max(lossavg), max(valavg))
    )
    return lossavg, valavg


# model = SimpleDLA(num_classes=1).to(device)
# if device == 'cuda':
#     model = torch.nn.DataParallel(model)

# model.load_state_dict(torch.load("./cifar/checkpoint/ckpt.pth")['net'], strict=False)


# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")
