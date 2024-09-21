from torchvision import transforms


# break image into 16 smaller images corresponding to given grid
def split_image(image):
    image_grid = [
        transforms.functional.crop(image, (i // 4) * 30, (i % 4) * 30, 30, 30)
        for i in range(16)
    ]
    return image_grid


# images_batch = split_image(gridpoints_dataset[0]["image"])
# image_grid = utils.make_grid(images_batch, nrow = 4)
# plt.imshow(image_grid.numpy().transpose((1, 2, 0)))
# plt.show(block=True)
