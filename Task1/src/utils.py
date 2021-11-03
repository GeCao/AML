from enum import Enum
import matplotlib.pyplot as plt


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2

def model_evaluation(computed_losses, epoch_step=200):
    n = len(computed_losses)
    epochs = [int(i * epoch_step) for i in range(n)]

    plt.figure(1)
    plt.title("train losses-epoch")
    plt.xlabel("epoch")
    plt.plot(epochs, computed_losses, 'g')
    plt.ylabel("loss")
    plt.ylim(0, 50)
    plt.show()