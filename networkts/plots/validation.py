import os

import numpy as np
import matplotlib.pyplot as plt


def plot_valid(
                file: str,
                save_path: str = None,
                ):
    f = open(file, "r")
    mae = []
    mape = []
    for row in f.readlines()[:-6]:
        score = row[:-1].split(' ')
        mape.append(float(score[0]))
        mae.append(float(score[1]))
    f.close()

    plt.figure(figsize=(18, 6))
    plt.plot(mape)
    plt.axhline(y=np.median(mape), color='r', linestyle='-')
    plt.xlabel("iteration")
    plt.ylabel("mape")
    plt.ylim(0, 2)
    plt.legend(['mape', 'median'])
    plt.title("Validation MAPE")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'mape.png'))

    plt.figure(figsize=(18, 6))
    plt.plot(mae)
    plt.axhline(y=np.median(mae), color='r', linestyle='-')
    plt.xlabel("iteration")
    plt.ylabel("mae")
    plt.legend(['mae', 'median'])
    plt.title("Validation MAE")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'mae.png'))
