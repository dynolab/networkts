import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_valid_score(
        file_name: str,
        title: str = None,
        fig_name: str = None,
        ):
    with open(file_name, 'rb') as f:
        score = pickle.load(f)
    mape = score['Mape']
    mae = score['Mae']

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(22, 14))
    if title is not None:
        fig.suptitle(title)
    ax[0].plot(mape, color='orange', label='mape')
    ax[0].axhline(
        y=np.median(mape),
        lw=1,
        color='r',
        linestyle='-',
        label='median'
        )
    ax[0].set_ylabel("mape")
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].set_title("Validation MAPE")

    ax[1].plot(mae, color='orange', label='mae')
    ax[1].axhline(
        y=np.median(mae),
        lw=1,
        color='r',
        linestyle='-',
        label='median'
        )
    ax[1].set_ylabel("mae")
    ax[1].set_xlabel("iteration")
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_title("Validation MAE")
    if fig_name is not None:
        plt.savefig(fig_name)
    return fig, ax


def plot_score_distribution_by_serie(
    serie: np.array or list,
    score_file: str,
    start_point: int,   # Abilene - 384, PeMSD7 - 172, Totem - 272
    test_size: int = 500,
    window: int = 1000,
    title: str = None,
    name: str = None
):
    with open(score_file, 'rb') as f:
        score = pickle.load(f)
    mape = score['Mape']
    mae = score['Mae']

    # Time point
    timesteps = [start_point + window]
    while timesteps[-1] < serie.shape[0]:
        if timesteps[-1]+test_size >= serie.shape[0]:
            break
        timesteps.append(timesteps[-1] + test_size)

    # Avg score
    n = mape.shape[0]//len(timesteps)
    mape = np.mean(mape.reshape(n, len(timesteps)), axis=0)
    mae = np.mean(mae.reshape(n, len(timesteps)), axis=0)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(24, 16))
    if title is not None:
        fig.suptitle(title)
    ax[0].plot(serie, color='b', label='real serie')
    ax[0].set_yscale('log')
    if mape is not None:
        ax[1].plot(timesteps, mape, 'o-', color='orange')
    if mae is not None:
        ax[2].plot(timesteps, mae, 'o-', color='orange')
    ax[2].set_yscale('log')
    ax[1].set_ylabel('mape')
    ax[2].set_ylabel('mae')
    ax[2].set_xlabel('Time, min')
    if name is not None:
        plt.savefig(name)
    return fig, ax


def plot_score_distribution_by_series_contour(
    score_file: str,
    df: pd.DataFrame,
    start_point: int,   # Abilene - 384, PeMSD7 - 172, Totem - 272
    test_size: int = 500,
    log: bool = True,
    window: int = 1000,
    title: str = None,
    fig_name: str = None,
):
    with open(score_file, 'rb') as f:
        score = pickle.load(f)
    score_mape = score['Mape']
    score_mae = score['Mae']

    # Time point
    x = [start_point + window]
    while x[-1] < df.shape[0]:
        if x[-1]+test_size >= df.shape[0]:
            break
        x.append(x[-1] + test_size)

    score_mape = np.mean(score_mape.reshape(df.shape[1], len(x)), axis=0)
    score_mae = np.mean(score_mae.reshape(df.shape[1], len(x)), axis=0)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(22, 9))
    Z = []
    for col in df.columns.values:
        if log:
            Z.append(np.log10(df[col].values))
        else:
            Z.append(df[col].values)
    cs = ax[0].contourf(Z, levels=30, extend='both')
    fig.colorbar(cs, ax=ax.ravel().tolist())
    ax[0].set_ylabel('Serie, №')
    ax[0].set_title('Time series contours')

    ax[1].plot(x, score_mape, 'o-', color='orange')
    ax[2].plot(x, score_mae, 'o-', color='orange')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[1].set_ylabel('mape')
    ax[2].set_ylabel('mae')
    ax[2].set_xlabel('Time, min')

    if title is not None:
        fig.suptitle(title)
    if fig_name is not None:
        plt.savefig(fig_name)
    return fig, ax


def plot_forecast(
        train: np.array or list,
        train_steps: np.array or list,
        test: np.array or list,
        test_steps: np.array or list,
        pred: np.array or list,
        title: str = None,
        name: str = None,
        ):
    plt.figure(figsize=(16, 8))
    if title is not None:
        plt.title(title)
    plt.plot(train_steps, train, label='train')
    plt.plot(test_steps, test, label='real')
    plt.plot(test_steps, pred, label='forecast', lw=1)
    plt.yscale('log')
    plt.legend()
    if name is not None:
        plt.savefig(name)


def plot_score_from_files(
        file_names: list,
        # file_names - list with names like as '.../score_xgb_{window_size}',
        # where window_size in [1000, 2000, ..., 5000, ...]
        title: str = None,
        fig_name: str = None,
        ):
    time = []
    score_avg = []
    score_med = []
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            score = pickle.load(f)
        time.append(score['Time'])
        score_avg.append([score['Avg_mape'], score['Avg_mae']])
        score_med.append([score['Mape_median'], score['Mae_median']])
    score_med = np.array(score_med)
    score_avg = np.array(score_avg)

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(24, 16))
    ax[0, 0].plot([1000*i for i in range(1, len(file_names)+1)], time, 'o-')
    ax[1, 0].plot(
                    [1000*i for i in range(1, len(file_names)+1)],
                    score_avg[:, 0],
                    'o-',
                    color='orange'
                    )
    ax[2, 0].plot(
                    [1000*i for i in range(1, len(file_names)+1)],
                    score_avg[:, 1],
                    'o-',
                    color='orange'
                    )
    ax[1, 1].plot(
                    [1000*i for i in range(1, len(file_names)+1)],
                    score_med[:, 0],
                    'o-',
                    color='orange'
                    )
    ax[2, 1].plot(
                    [1000*i for i in range(1, len(file_names)+1)],
                    score_med[:, 1],
                    'o-',
                    color='orange'
                    )
    ax[2, 0].set_yscale('log')
    ax[2, 1].set_yscale('log')
    ax[2, 0].set_xlabel('размер окна')
    ax[2, 1].set_xlabel('размер окна')
    ax[0, 0].set_ylabel('время, сек')
    ax[1, 0].set_ylabel('mape')
    ax[2, 0].set_ylabel('mae')
    ax[0, 0].set_title('Avg')
    ax[0, 1].set_title('Median')
    if title is not None:
        fig.suptitle(title)
    if fig_name is not None:
        plt.savefig(fig_name)
    return fig, ax


def bar_plot(
        ax,
        data,
        colors=None,
        total_width=0.8,
        single_width=1,
        legend=True,
        fig_name=None
        ):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names
        of the data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[i % len(colors)]
                )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
    if fig_name is not None:
        plt.savefig(fig_name)
