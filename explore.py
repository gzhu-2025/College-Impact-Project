import pandas as pd
import os
import torch
import torch.optim as optim
import time

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

optimizer = torch.optim.Adam(net.parameters())
criterion = torch.nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1000, T_mult=1, eta_min=1e-4)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X / 255.0, y, test_size=0.2)

import numpy as np

losses = []
total_train_acc = []
total_val_acc = []
val_losses = []

batch_size = 8
def train(epochs):
    for step in range(epochs):
        net.train()

        batch = torch.randperm(X_train.shape[0])[:batch_size]
        X_batch = X_train[batch].to(device)
        y_batch = y_train[batch].to(device)

        optimizer.zero_grad()

        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        train_acc = (outputs.argmax(1) == y_batch).float().mean()
        total_train_acc.append(train_acc.item())

        if step % 10 == 0:
            cat_correct = {cat[0]: 0 for cat in cat_to_int.items()}
            # print(cat_correct)
            net.eval()
            with torch.no_grad():
                val_output = net(X_test.to(device))
                # compute val accuracy
                val_acc = (val_output.argmax(1) == y_test.to(device)).float().mean()
                total_val_acc.append(val_acc.item())
                val_loss = criterion(val_output, y_test.to(device))
                val_losses.append(val_loss.item())
                os.system('cls') if os.name == 'nt' else os.system('clear')
                
                print(f"Epoch: {step}")
                print(f"Train Loss: {np.mean(losses[-50:]):.2f}")
                print(f"Train Accuracy: {np.mean(total_train_acc[-50:]):.2f}")
                print(f"Val Loss: {np.mean(val_losses[-50:]):.2f}")
                print(f"Val Accuracy: {np.mean(total_val_acc[-50:]):.2f}")
                

                

                for j in range(len(cat_correct)):
                    cat = int_to_cat.get(j)

                    X_class = X_test[y_test == j]                    
                    out_class = net(X_class.to(device))

                    cat_correct[cat] += (out_class.argmax(1) == j).int().sum().item()
                    print(f"Accuracy for class {cat} {' '*(13-len(int_to_cat.get(j)))}: {cat_correct.get(int_to_cat.get(j))/(y_test == j).float().sum().item():.2f} ({cat_correct.get(int_to_cat.get(j))}/{(y_test == j).float().sum().item()})")
                    
                print(f"Macro Accuracy: {sum([b for a, b in cat_correct.items()])/y_test.shape[0]:.2f}")

    plot_loss(
        net._get_name(), losses, epochs, val_losses, max(max(losses), max(val_losses))
    )
epochs=6000
train(epochs=epochs)
torch.save(net.state_dict(), f"states/{net._get_name()}_{epochs}.pth")