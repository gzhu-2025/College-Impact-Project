import torch
from torch.utils.data import DataLoader

from datasets import *
from utils import *
from transforms import *
from models import *
from model_actions import *

# model = LeNet()

# X = torch.randn(1, 3, 32, 32)

# model(X).shape 

# sd = model.state_dict()

# for k, v in sd.items():
#     print(k, v.shape)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

model2 = LeNet2().to(device)
model2.load_state_dict(torch.load("./checkpoints/lenet.pth")['net'], strict=False)


# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lr = 1e-4

optimizer = torch.optim.AdamW(model2.parameters(), lr=lr)

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


# avgloss, losses = test(training_dataloader, model2, loss_fn)
# print(f"average test loss of dataset: {avgloss}\n")

from sklearn.model_selection import train_test_split

range_train, range_test = train_test_split(range(len(gridpoints_dataset)), test_size=0.2)

train_dataset = torch.utils.data.Subset(gridpoints_dataset, range_train)
val_dataset = torch.utils.data.Subset(gridpoints_dataset, range_test)

# # val_dataset = GridPointsDataset(
# #     category="Crosswalk",
# #     array_dir="data/labels/_ndarrays",
# #     image_dir="data/images",
# #     train=False,
# #     seed=seed,
# #     transform=transforms.Compose([
# #         ToTensor(),
# #         ColorChannelCorrection(),
# #         ]),
# # )

training_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)   
show_batch(training_dataloader, train=True)

# # print(len(gridpoints_dataset), len(val_dataset))
# # print(seed)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

# model = NeuralNetwork().to(device)
# # print(model)

# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lr = 1e-4

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

train_set_testing = set([x['image_name'] for x in train_dataset])
val_set_testing = set([x['image_name'] for x in val_dataset])


assert train_set_testing.intersection(val_set_testing) == set(), "Train and validation sets are not disjoint"

lossavg, valavg  = train(training_dataloader, val_dataloader, model2, loss_fn, optimizer, epochs=100)

