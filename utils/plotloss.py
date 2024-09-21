import matplotlib.pyplot as plt
import numpy as np

def plot_loss(title, lossavg, epochs, valavg, maxloss):
    plt.figure()
    plt.ioff()

    
    newval = np.interp(
                       np.linspace(0, epochs, epochs), 
                       np.linspace(0, epochs, len(valavg)), 
                       valavg
                       )
    plt.plot(lossavg, label='train loss')
    plt.plot(newval, label='val loss')
    
    plt.title(f"{title}, Epochs: {epochs}")
    plt.axis((0, epochs, 0, maxloss))
    # print(max(loss))
    plt.legend()
    plt.savefig(f"./plots/{title}_{epochs}.png")

    plt.show(block=True)
     