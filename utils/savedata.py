import os, torch

image_directories = os.listdir("data")

for dir in image_directories:
    print(dir, ": ", len(os.listdir(f"data/{dir}")))


cat_to_int = {}
def saveData():
    current_int = 0
    X = []
    y = []
    for dir in image_directories:
        current_direcotry = f"data/{dir}"
        current_category = dir
        cat_to_int[current_category] = current_int

        current_images = []
        for im in os.listdir(current_direcotry):
            im = io.imread(f"{current_direcotry}/{im}")
            if im.shape[0] == 120 and im.shape[1] == 120:
                current_images.append(torch.tensor(im[:, :, :3]))  # type: ignore

        X.append(torch.stack(current_images))
        y.extend([current_int] * X[-1].shape[0])
        current_int += 1
    X = torch.cat(X, dim=0)
    y = torch.tensor(y)

    torch.save(X, "X.pth")
    torch.save(y, "y.pth")
    return cat_to_int
