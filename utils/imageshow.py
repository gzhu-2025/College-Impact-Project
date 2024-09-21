import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
