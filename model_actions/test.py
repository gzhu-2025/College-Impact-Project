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
    best_acc = 0
    # size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()
    valloss = []
    valavg = []

    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):

            image, grid = sample["image"].to(device), sample["grid"].to(device)
            i = 0
            for image_square in split_image(image):

                pred = model(image_square)
                grid_val = grid[0][i].unsqueeze(0).unsqueeze(0)
            
                loss = loss_fn(pred, grid_val.type(torch.float32)) 

                valloss.append(loss.item())
                valavg.append(np.mean(valloss[-50:]))

                _, predicted = pred.max(1)
                total += 1
                correct += predicted.eq(grid_val).sum().item()
                
                # os.system('cls')
                # print(f"batch: {batch}\nImage: [{i + 1}/16]\nTest Loss: {loss:>f}\nAverage Loss: {np.mean(valloss[-50:])}")

                i += 1

    acc = 100. * correct/total
    if acc > best_acc:
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(model.state_dict(), f'./checkpoints/{model._get_name()}.pth')
        best_acc = acc
        
    

    return np.mean(valloss), valloss, acc



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
