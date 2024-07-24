import torch

import numpy as np

from utils import *


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# model = NeuralNetwork().to(device)
# print(model)

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)




def test(dataloader, model, loss_fn):
    # size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()
    valloss = []
    
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):

            image, grid = sample["image"].to(device), sample["grid"].to(device)
            i = 0
            for image_square in split_image(image):

                pred = model(image_square)
                grid_val = grid[0][i].unsqueeze(0).unsqueeze(0)
            
                loss = loss_fn(pred, grid_val.type(torch.float32)) 

                valloss.append(loss.item())
                
                i += 1
    return np.mean(valloss)



# train(training_dataloader, model, loss_fn, optimizer, epochs=250)

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))

# test(training_dataloader, model, loss_fn)

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
