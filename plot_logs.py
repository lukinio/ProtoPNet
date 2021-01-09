import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from torch import Tensor

def plot_logs(prefix, acc, *loss):

    x = np.arange(1, len(acc)+1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    ax.plot(x, acc, label="accuracy")
    ax.set_xlabel('epochs')
    ax.set_ylabel("accuracy")
    ax.legend(loc="best")
    fig.savefig(prefix+'accuracy.png', dpi=fig.dpi)

    names = ["total loss", "cross entropy", "cluter cost", "separation cost"]
    x = np.arange(1, len(loss[0])+1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    for i, arr in enumerate(loss):
        ax.plot(x, arr, label=names[i])
    ax.set_xlabel('epochs')
    ax.legend(loc="best")
    fig.savefig(prefix+'loss.png', dpi=fig.dpi)



def plot_conf_matrix(conf_matrix, epoch, path):
    fig = plt.figure()
    conf_matrix = np.array(conf_matrix)
    normlized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(normlized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    
    thresh = normlized.max() / 2.
    for i, j in product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(normlized[i, j], '.6f')+"%",
                    verticalalignment="top",
                    horizontalalignment="center",
                    color="white" if normlized[i, j] > thresh else "black")              
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    color="white" if normlized[i, j] > thresh else "black")
    
    plt.xticks([])
    plt.yticks([])
    tick_marks = np.arange(conf_matrix.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('Actual')
    plt.xlabel('Predict')
    plt.title("epoch "+str(epoch))
    plt.colorbar()
    fig.savefig(path+str(epoch)+'_conf_matrix.png', dpi=fig.dpi)


if __name__ == "__main__":
    test_acc = [0.5, 0.6, 0.7, 0.8, 0.9]
    test_cross_ent = [0.5, 0.4, 0.3, 0.2, 0.1]
    test_cluster_cost = [0.6, 0.5, 0.4, 0.3, 0.2]
    test_sep_cost = [0.7, 0.6, 0.5, 0.4, 0.3]
    test_total_loss = [1.8, 1.5, 1.2, 0.9, 0.6]

    X = [[20, 1], [2, 22]]
    X = [[265, 268], [235, 232]]
    plot_conf_matrix(X, 1, "")

    # plot_logs("10test_", test_acc, test_total_loss, test_cross_ent, test_cluster_cost, test_sep_cost)