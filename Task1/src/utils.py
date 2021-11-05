from enum import Enum
import matplotlib.pyplot as plt


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2

def model_evaluation(computed_losses, train_losses, epoch_step=200):
    n = len(computed_losses)
    epochs = [int(i * epoch_step) for i in range(n)]

    plt.figure(1)
    plt.title("train losses-epoch")
    plt.xlabel("epoch")
    plt.plot(epochs, computed_losses, 'orange', label="test")
    plt.plot(epochs, train_losses, 'b', label="train")
    plt.ylabel("loss")
    plt.ylim(0, 100)
    plt.legend()
    plt.show()