import os
import warnings
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sttf.utils import common
sys.path.append(common.CONF['directory']['path_networks'])
from decompositions.basic import log_target, inverse_log_target
from decompositions.basic import exp_smoth, moving_avg, nothing


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    plt.style.use('config/default.mplstyle')

    df = pd.read_csv(
            os.path.join(
                os.getcwd(),
                common.CONF['datasets']['totem']['root'],
                common.CONF['datasets']['totem']['edges_traffic']
                ),
            index_col=0,     # abilene, totem
            # header=None,   # pemsd7
            )

    for el in df.columns.values:
        df.loc[df[el] < 1000, el] = 1000

    train_size = 4000
    test_size = 500
    period = 96
    delta_time = 15
    feature = df.columns.values[44]
    ind = np.array([el*delta_time for el in range(df.shape[0])])

    '''
    # plot validation score
    from sttf.plots.validation import plot_valid_score
    fig, ax = plot_valid_score('valid_results/Totem/window/es/score_ar_3000')
    plt.show()
    '''
    '''
    # plot score distribution by serie
    from sttf.plots.validation import plot_score_distribution_by_serie
    fig, ax = plot_score_distribution_by_serie(
        df[feature].values,
        'valid_results/Totem/window/es/score_ar_3000',
        272,
        500,
        3000,
        f'AR, Window size = 3000',
    )
    ax[0].plot(moving_avg(df[feature].values, [30]), color='g', label='m_a')
    ax[0].legend()
    plt.show()
    '''
    '''
    from sttf.plots.validation import plot_score_distribution_by_series_contour
    fig, ax = plot_score_distribution_by_series_contour(
        'valid_results/Abilene/window/ssa/score_ar_3000',
        df,
        384,
        True,
        3000,
        'Abilene, SSA, AR, Window size = 3000 - score distribution',
    )
    #plt.savefig('pic/Totem/valid/window/ssa/totem_score_distribution.png', dpi=200)
    plt.show()
    '''
    '''
    # plot forecast for one serie
    from sttf.plots.validation import plot_forecast
    x = ind/15
    y = df[feature].values
    plot_forecast(
        y[771:4770],
        x[771:4770],
        y[4771:5270],
        x[4771:5270],
        pred,
        f'Window = {train_size}, score = {mape(y_pred=pred, y_true=y[4771:5270]):.3f}, {mae(pred, y[4771:5270]):.3f}',
        )
    plt.show()
    '''

    '''
    # plot score from files
    from sttf.plots.validation import plot_score_from_files
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
    '''

    '''
    # plot bar distribution
    from sttf.plots.validation import bar_plot
    data = {}
    for train_size in [1000, 2000, 3000, 4000, 5000]:
        with open(f'valid_results/PeMSD7/window/score_ar_{train_size}', 'rb') as f:
            score = pickle.load(f)
        score_mae = np.array(score['Mae'])
        score_mape = np.array(score['Mape'])
        score_mape = score_mape.reshape(228, score_mape.shape[0]//228)
        score_mae = score_mae.reshape(228, score_mae.shape[0]//228)
        score_mape = [np.mean(el) for el in score_mape]
        score_mae = [np.mean(el) for el in score_mae]
        data[train_size] = score_mape

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('AutoReg Mape for PeMSD7')
    bar_plot(ax, data, total_width=.8, single_width=1)
    plt.show()
    '''
