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

model = NeuralNetwork().to(device)
# print(model)

# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lr = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

training_dataloader = DataLoader(gridpoints_dataset, batch_size=1, shuffle=True, num_workers=0)

show_batch(training_dataloader, train=True)


model2 = LeNet2().to(device)
model2.load_state_dict(torch.load("./checkpoints/lenet.pth")['net'], strict=False)


loss = test(training_dataloader, model2, loss_fn)

os.system('cls')
print(f"Test Loss: {loss:>f}")

