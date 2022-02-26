import os
import matplotlib.pyplot as plt # for plotting

# this file will contain utility functions

def normalization(imgs, labels):
    return imgs/255, labels/255

def data_plotting(iters, losses):
    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    # plt.title("Training Curve")
    # plt.plot(iters, train_acc, label="Training")
    # plt.plot(iters, val_acc, label="Validation")
    # plt.xlabel("Iterations")
    # plt.ylabel("Validation Accuracy")
    # plt.legend(loc='best')
    # plt.show()

    # train_acc.append(get_accuracy(model, train=train_type))
    # print("Final Training Accuracy: {}".format(train_acc[-1]))
    # print("Final Validation Accuracy: {}".format(val_acc[-1]))