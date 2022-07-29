import os
import warnings
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call

import networkts


LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}')
    warnings.filterwarnings('ignore')

    dataset = call(cfg.dataset)    
    df = dataset.edge_timeseries.data

    # setting the low limit for abilene's, totem's traffic
    if dataset.name in ['Abilene', 'Totem']:
        for feature in df.columns.values:
            df.loc[df[feature] < 1000, feature] = 1000

    train_size = 4000
    test_size = 500
    period = dataset.period
    delta_time = dataset.delat_time
    feature = df.columns.values[44]
    ind = df.index.values

    # plot validation score
    from networkts.plots.validation import plot_valid_score
    fig, ax = plot_valid_score('valid_results/Totem/window/es/score_ar_3000')
    plt.show()

    # plot score distribution by serie
    decomposition = instantiate(cfg.decomposition)
    from networkts.plots.validation import plot_score_distribution_by_serie
    fig, ax = plot_score_distribution_by_serie(
        df[feature].values,
        'valid_results/Totem/window/es/score_ar_3000',
        272,
        500,
        3000,
        f'AR, Window size = 3000',
    )
    ax[0].plot(
        decomposition.transform(df[feature].values),
        color='g',
        label=decomposition.name
        )
    ax[0].legend()
    plt.show()

    # plot score distribution by serie with contour
    from networkts.plots.validation import plot_score_distribution_by_series_contour
    fig, ax = plot_score_distribution_by_series_contour(
        score_file='valid_results/Totem/window/ssa/score_ar_3000',
        df=df,
        start_point=272,
        test_size=500,
        log=True,
        window=3000,
        title='Abilene, SSA, AR, Window size = 3000 - score distribution',
    )
    # plt.savefig('pic/Totem/valid/window/ssa/totem_score_distribution.png', dpi=200)
    plt.show()

    # plot score from files
    from networkts.plots.validation import plot_score_from_files
    data = 'Totem'
    for smooth in ['log', 'es', 'ssa']:
        for method in ['ar', 'xgb']:
            names_list = []
            for train_size in [1000, 2000, 3000, 4000, 5000]:
                names_list.append(f'valid_results/{data}/window/{smooth}/score_{method}_{train_size}')

            title = f'{data}, {smooth}, {method}'
            fig_name = f'pic/{data}/valid/window/{smooth}/score_{data}_{smooth}_{method}.png'
            plot_score_from_files(names_list, title, fig_name)
    plt.show()
 
    # plot bar distribution
    from networkts.plots.validation import bar_plot
    data = {}
    for train_size in [1000, 2000, 3000, 4000, 5000]:
        with open(f'valid_results/Totem/window/log/score_ar_{train_size}', 'rb') as f:
            score = pickle.load(f)
        score_mae = np.array(score['Mae'])
        score_mape = np.array(score['Mape'])
        n = df.shape[1]
        score_mape = score_mape.reshape(n, score_mape.shape[0]//n)
        score_mae = score_mae.reshape(n, score_mae.shape[0]//n)
        score_mape = [np.mean(el) for el in score_mape]
        score_mae = [np.mean(el) for el in score_mae]
        data[train_size] = score_mape

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('AutoReg Mape for Totem with log')
    bar_plot(ax, data, total_width=.8, single_width=1)
    plt.show()
