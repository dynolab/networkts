import os

import numpy as np
import matplotlib.pyplot as plt


def plot_traffic(
        serie,
        title=None,
        name=None,
        ):
    fig, ax = plt.subplots(figsize=(16, 8))
    if title is not None:
        fig.suptitle(title)
    ax.plot(serie, color='b', label='real serie')
    ax.set_yscale('log')
    ax.set_xlabel('time, min')
    ax.set_ylabel('traffic')
    if name is not None:
        plt.savefig(name)
    return fig, ax
