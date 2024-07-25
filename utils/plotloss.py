import matplotlib.pyplot as plt

def plot_loss(title, lossavg, epochs, valavg, maxloss):
    plt.figure()
    plt.ioff()


    plt.plot(lossavg, label='average loss')
    plt.plot(valavg, label='average val loss')
    
    plt.title(f"{title}, Epochs: {epochs}")
    plt.axis((0, epochs, 0, maxloss))
    # print(max(loss))
    plt.legend()
    plt.savefig(f"./plots/{title}_{epochs}.png")

    plt.show(block=True)
     