import pandas as pd
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


def plot_dataset_contour(
    df: pd.DataFrame,
    log: bool = True,
    title: str = None,
    fig_name: str = None,
):
    Z = []
    for col in df.columns.values:
        if log:
            Z.append(np.log10(df[col].values))
        else:
            Z.append(df[col].values)

    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contourf(Z, levels=30, extend='both')
    fig.colorbar(cs, shrink=0.8)
    ax.set_xlabel('Time, 5 min')
    ax.set_ylabel('Serie, №')
    if title is not None:
        plt.title(title)
    if fig_name is not None:
        plt.savefig(fig_name)
    return fig, ax
