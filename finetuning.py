import torch
from torchvision.transforms._presets import ImageClassification
import torchvision
from PIL import Image
import glob
import os
import numpy as np


weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES

# weights.transforms
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py
transform = ImageClassification(
    crop_size=224,
    mean=(0.48235, 0.45882, 0.40784),
    std=(0.00392156862745098, 0.00392156862745098, 0.00392156862745098),
)


model = torchvision.models.vgg16(weights=weights)

# number of weights:

print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")


file_paths = glob.glob("data/images/" + "**/*.png")
print("Number of images:", len(file_paths))

model.eval()

# free gradients
for param in model.parameters():
    param.requires_grad = False

# open an image


im = Image.open(file_paths[0])
result = model.features(transform(im).unsqueeze(0))
print(result.shape)

transformed_images = []
categories = []
for file_path in file_paths:
    with Image.open(file_path) as im:
        # remove alpha channel
        if im.mode == "RGBA":
            im = im.convert("RGB")
        transformed_images.append(transform(im).unsqueeze(0))
        categories.append(os.path.normpath(file_path).split("\\")[2])
transformed_images = torch.cat(transformed_images, dim=0)


total_cats = list(set(categories))


def cat_to_idx(cat):
    return total_cats.index(cat)


y = torch.tensor([cat_to_idx(cat) for cat in categories])


# train test split random 80 - 20
# 80% for training and 20% for validation
inds = torch.randperm(transformed_images.size(0))
train_size = int(0.8 * transformed_images.size(0))
train_inds = inds[:train_size]
val_inds = inds[train_size:]

train_images = transformed_images[train_inds]
val_images = transformed_images[val_inds]
train_y = y[train_inds].long()
val_y = y[val_inds].long()

print(train_images.shape)
print(val_images.shape)


class FinetuneModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(25088, 13)

    def forward(self, x: Image) -> torch.Tensor:
        x = x.to(self.model.features[0].weight.device)
        x = self.model.features(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # flatten x
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")
finetunemodel = FinetuneModel(model).to(device)
print(finetunemodel(train_images[:3]))


# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.AdamW(finetunemodel.parameters(), lr=3e-4)

# training loop
batch_size = 32
losses = []


for epoch in range(10):
    finetunemodel.train()
    for i in range(0, len(train_images), batch_size):
        optimizer.zero_grad()
        x = train_images[i : i + batch_size].to(device)
        y_pred = finetunemodel(x)
        y_true = train_y[i : i + batch_size].to(device)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    finetunemodel.eval()
    with torch.no_grad():
        counter = 0
        val_loss = 0
        for i in range(0, len(val_images), batch_size):
            x = val_images[i : i + batch_size].to(device)
            y_pred = finetunemodel(x)
            y_true = val_y[i : i + batch_size].to(device)
            y_guess = torch.argmax(y_pred, dim=1)
            counter += (y_guess == y_true).float().sum()
            val_loss += loss_fn(y_pred, y_true)
        accuracy = counter / val_images.size(0)
        print(
            f"val accuracy: {accuracy.item():.2f} val loss: {val_loss.item()/val_images.size(0):.2f}"
        )
        print(f"epoch = {epoch} train loss={np.mean(losses[-100:]):.2f}")


# save model
torch.save(finetunemodel.state_dict(), "model.pt")

val_pred = []
for i in range(0, len(val_images), batch_size):
    x = val_images[i : i + batch_size].to(device)
    y_guess = torch.argmax(finetunemodel(x), dim=1)
    val_pred.extend(y_guess.cpu().tolist())
val_pred = torch.tensor(val_pred).long()

# compute accuracy along each category:

for cat in total_cats:
    cat_inds = val_y == cat_to_idx(cat)  # where is val_y equal to cat
    cat_acc = (val_pred[cat_inds] == val_y[cat_inds]).float().mean()
    print(f"{cat}, accuracy: {cat_acc.item():.2f} Data Points: {cat_inds.sum()}")
