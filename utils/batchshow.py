import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import utils

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
