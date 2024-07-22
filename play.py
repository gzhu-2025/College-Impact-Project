from models import LeNet

model = LeNet()

import torch

X = torch.randn(1, 3, 32, 32)

model(X).shape 

sd = model.state_dict()

for k, v in sd.items():
    print(k, v.shape)





"""LeNet in PyTorch."""
import torch.nn as nn
import torch.nn.functional as F


class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, 10)

        self.crosswalk = nn.Linear(84, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        # out = self.fc3(out)
        out = self.crosswalk(out)
        return out


model2 = LeNet2()
model2.load_state_dict(model.state_dict(), strict=False)
