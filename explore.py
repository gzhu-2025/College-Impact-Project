import pandas as pd
import os
import torch

from utils import plot_loss



# sum = 0
# for dir in image_directories:
#     print(dir + ":", len(os.listdir(f"data/images/{dir}")))
#     sum += len(os.listdir(f"data/images/{dir}"))

# print("total:", sum)


cat_to_int = {
              'Chimney': 0, 
              'Stair': 1, 
              'Traffic Light': 2, 
              'Motorcycle': 3, 
              'Crosswalk': 4, 
              'Palm': 5, 
              'Bicycle': 6, 
              'Car': 7, 
              'Bridge': 8, 
              'Hydrant': 9, 
              'Bus': 10
              }
int_to_cat = {v: k for k, v in cat_to_int.items()}


from models.lenet import LeNet

X, y = torch.load('X.pt'), torch.load('y.pt')

net = LeNet()
# mps device
device = "cuda" if torch.cuda.is_available() else "cpu"


net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X / 255.0, y, test_size=0.2)

import numpy as np

losses = []
val_losses = []
batch_size = 8
for step in range(500):
    net.train()
    batch = torch.randperm(X_train.shape[0])[:batch_size]
    X_batch = X_train[batch].to(device)
    y_batch = y_train[batch].to(device)
    optimizer.zero_grad()
    outputs = net(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    train_acc = (outputs.argmax(1) == y_batch).float().mean()

    if step % 10 == 0:
        net.eval()
        with torch.no_grad():
            val_output = net(X_test.to(device))
            # compute val accuracy
            val_acc = (val_output.argmax(1) == y_test.to(device)).float().mean()
            val_loss = criterion(val_output, y_test.to(device))
            val_losses.append(val_loss.item())
            print(f"Val Accuracy: {val_acc.item():.2f}")
            print(f"Val Loss: {val_loss.item():.2f}")
            print(f"Train Loss: {np.mean(losses[-50:]):.2f}")
            print(f"Train Accuracy: {train_acc.item():.2f}")
            accs_by_class = []
            for j in range(11):
                X_class = X_test[y_test == j]
                
                total = X_class.shape[0]
                out_class = net(X_class.to(device))
                # accuracy along this class
                acc_class = (out_class.argmax(1) == j).float().mean().item()
                print(f"Accuracy for class {int_to_cat[j]}: {acc_class:.2f}")
                accs_by_class.append(acc_class)
            print(f"Macro Accuracy: {np.mean(accs_by_class):.2f}")

plot_loss(
    net._get_name(), losses, 500, val_losses, max(max(losses), max(val_losses))
)
