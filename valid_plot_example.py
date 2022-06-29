import os
import warnings
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from datetime import datetime
from sklearn.compose import TransformedTargetRegressor

from sttf.utils import common
sys.path.append(common.CONF['directory']['path_networks'])
from forecasters.xgboost import NtsXgboost
from utils.sklearn_helpers import SklearnWrapperForForecaster
from utils.sklearn_helpers import build_target_transformer
from cross_validation import ValidationBasedOnRollingForecastingOrigin as Valid
from forecasters.autoreg import NtsAutoreg
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

    df = df.replace([0], 0.1)

    train_size = 4000
    test_size = 500
    period = 96
    delta_time = 15
    feature = df.columns.values[44]

    ind = np.array([el*delta_time for el in range(df.shape[0])])

    score_mape = []
    score_mae = []
    time = datetime.now()

    cross_val = Valid(
            n_test_timesteps=test_size,
            n_training_timesteps=train_size,
            n_splits=df.shape[0]//test_size - train_size//test_size,
            max_train_size=np.Inf
            )

    # XGB
    
    model = build_target_transformer(
                TransformedTargetRegressor,
                SklearnWrapperForForecaster(NtsXgboost()),
                func=log_target,
                inverse_func=inverse_log_target,
                params=None,
                inverse_params=None,
                )
    

    # AR
    '''
    model = build_target_transformer(
                TransformedTargetRegressor,
                SklearnWrapperForForecaster(
                    NtsAutoreg(
                        lags=3,
                        seasonal=True,
                        period=period
                        )),
                func=log_target,
                inverse_func=inverse_log_target,
                params=None,
                inverse_params=None,
                )
    '''

    model.fit(y=df[feature].values[771:4770], X=ind[771:4770])
    pred = model.predict(ind[4771:5270])
    '''
    t = cross_val.evaluate(
            forecaster=model,
            y=moving_avg(df[feature].values, 30),
            X=ind
            )

    t = np.array(t)
    t1, t2 = t[:, 0], t[:, 1]
    score_mape += t1.tolist()
    score_mae += t2.tolist()
    print(np.mean(score_mape))
    '''


    # plot_score_distribution_by_series
    from sttf.plots.validation import plot_score_distribution_by_series
    x = [
        4272,
        4772,
        5272,
        5772,
        6272,
        6772,
        7272,
        7772,
        8272,
        8772,
        9272,
        9772,
        10272,
        ]
    fig, ax = plot_score_distribution_by_series(
        df[feature].values,
        x,
        score_mape,
        score_mae,
        f'XGB, Window size = {train_size}',
    )
    ax[0].plot(moving_avg(df[feature].values, 30), color='g', label='m_a')
    ax[0].legend()
    plt.show()
    

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
    

    # plot_score_from_files
    from sttf.plots.validation import plot_score_from_files
    file_names = []
    for train_size in [1000*i for i in range(1, 6)]:
        file_names.append(
            f'valid_results/PeMSD7/window/score_xgb_{train_size}'
            )

    fig, ax = plot_score_from_files(
        file_names
        )
    plt.show()
    
    
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
