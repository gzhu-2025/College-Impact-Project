import matplotlib.pyplot as plt

def plot_loss(lossavg, epochs, valavg, maxloss):
    plt.figure()
    plt.ioff()


    plt.plot(lossavg, label='average loss')
    plt.plot(valavg, label='average val loss')

    plt.axis((0, epochs, 0, maxloss))
    # print(max(loss))
    plt.legend()
    plt.show(block=True)
     