from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


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
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def plot_stats(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 5))
    axs[0][0].hist(means, bins=100)
    axs[0][0].set_title('Means')
    axs[0][1].hist(stds, bins=100)
    axs[0][1].set_title('Stand divs.')
    axs[1][0].hist(maxs, bins=100)
    axs[1][0].set_title('Max')
    axs[1][1].hist(mins, bins=100)
    axs[1][1].set_title('Min')

    fig.tight_layout(pad=1.0)


def plot_classes(cls0_X, cls1_X, cls2_X, cls3_X):
    mean0, mean1, mean2, mean3 = np.mean(cls0_X, axis=0), np.mean(cls1_X, axis=0), np.mean(cls2_X, axis=0), np.mean(cls3_X, axis=0)
    std0, std1, std2, std3 = np.std(cls0_X, axis=0), np.std(cls1_X, axis=0), np.std(cls2_X, axis=0), np.std(cls3_X, axis=0)
    max0, max1, max2, max3 = np.max(cls0_X, axis=0), np.max(cls1_X, axis=0), np.max(cls2_X, axis=0), np.max(cls3_X, axis=0)
    min0, min1, min2, min3 = np.min(cls0_X, axis=0), np.min(cls1_X, axis=0), np.min(cls2_X, axis=0), np.min(cls3_X, axis=0)

    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(8, 6))
    axs[0][0].hist(mean0, bins=100, color='c')
    axs[0][0].set_title('Mean class 0')
    axs[0][1].hist(mean1, bins=100, color='m')
    axs[0][1].set_title('Mean class 1')
    axs[0][2].hist(mean2, bins=100, color='y')
    axs[0][2].set_title('Mean class 2')
    axs[0][3].hist(mean3, bins=100, color='b')
    axs[0][3].set_title('Mean class 3')

    axs[1][0].hist(std0, bins=100, color='c')
    axs[1][0].set_title('Std class 0')
    axs[1][1].hist(std1, bins=100, color='m')
    axs[1][1].set_title('Std class 1')
    axs[1][2].hist(std2, bins=100, color='y')
    axs[1][2].set_title('Std class 2')
    axs[1][3].hist(std3, bins=100, color='b')
    axs[1][3].set_title('Std class 3')

    axs[2][0].hist(max0, bins=100, color='c')
    axs[2][0].set_title('Max class 0')
    axs[2][1].hist(max1, bins=100, color='m')
    axs[2][1].set_title('Max class 1')
    axs[2][2].hist(max2, bins=100, color='y')
    axs[2][2].set_title('Max class 2')
    axs[2][3].hist(max3, bins=100, color='b')
    axs[2][3].set_title('Max class 3')

    axs[3][0].hist(min0, bins=100, color='c')
    axs[3][0].set_title('Min class 0')
    axs[3][1].hist(min1, bins=100, color='m')
    axs[3][1].set_title('Min class 1')
    axs[3][2].hist(min2, bins=100, color='y')
    axs[3][2].set_title('Min class 2')
    axs[3][3].hist(min3, bins=100, color='b')
    axs[3][3].set_title('Min class 3')

    fig.tight_layout(pad=1.0)