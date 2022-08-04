import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae

from networkts.toy_models import var3_generator, var3_generator_with_season
from networkts.utils.create_dummy_vars import create_dummy_vars
from networkts.forecasters.autoreg import NtsAutoreg
from networkts.forecasters._var import NtsVar
from networkts.plots.validation import *
from networkts.utils.convert_time import time
from networkts.utils.create_features import create_features


if __name__ == '__main__':
    series_size = 2500
    x, y, z, m = var3_generator_with_season(series_size, 288)
    data = pd.DataFrame(np.array([x, y, z, m]).T, columns=['x', 'y', 'z', 'm'], index=range(series_size))
    period = 288
    maxlags = 300

    dummy_vars_no_period = data.index#create_features([time(el) for el in data.index.values])
    dummy_vars = create_dummy_vars(data.index, period)

    train_size = 2000
    
    # AR with period
    pred = []
    insample = []
    for feature in data.columns:
        model_ar = NtsAutoreg(
            lags=7,
            seasonal=True,
            period=period,
            trend='n',
            cov_type='HAC',
            cov_kwds={'maxlags': maxlags},
        ).fit(
            y=data[feature].values[:train_size],
            X=dummy_vars.values[:train_size],
            )

        insample.append( model_ar.insample(X=dummy_vars.values[:train_size]))
        pred.append(model_ar.predict(X=dummy_vars.values[train_size:]))

    insample = np.array(insample)
    pred = np.array(pred)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for AR in-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[:train_size], insample[j, :], label='in-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[7:train_size, j], insample[j, 7:]):.3f}, '
            f'mae: {mae(data.iloc[7:train_size, j], insample[j, 7:]):.3f}'
        )
    plt.savefig('tests/figs/insample_ar_with_season.png', dpi=200)
    
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for AR out-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[train_size:], data.iloc[train_size:, j], label='real')
        ax[j].plot(data.index[train_size:], pred[j], label='out-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[train_size:, j], pred[j]):.3f}, '
            f'mae: {mae(data.iloc[train_size:, j], pred[j]):.3f}'
        )
    plt.savefig('tests/figs/outsample_ar_with_season.png', dpi=200)
    '''
    # AR without period
    pred = []
    insample = []
    for feature in data.columns:
        model_ar = NtsAutoreg(
            lags=7,
            seasonal=False,
            trend='n',
            cov_type='HAC',
            cov_kwds={'maxlags': maxlags},
        ).fit(
            y=data[feature].values[:train_size],
            X=dummy_vars_no_period.values[:train_size],
            )

        insample.append( model_ar.insample(X=dummy_vars_no_period.values[:train_size]))
        pred.append(model_ar.predict(X=dummy_vars_no_period.values[train_size:]))

    insample = np.array(insample)
    pred = np.array(pred)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for AR in-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[:train_size], insample[j, :], label='in-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[7:train_size, j], insample[j, 7:]):.3f}, '
            f'mae: {mae(data.iloc[7:train_size, j], insample[j, 7:]):.3f}'
        )
    plt.savefig('tests/figs/insample_ar_without_period.png', dpi=200)
    
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for AR out-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[train_size:], data.iloc[train_size:, j], label='real')
        ax[j].plot(data.index[train_size:], pred[j], label='out-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[train_size:, j], pred[j]):.3f}, '
            f'mae: {mae(data.iloc[train_size:, j], pred[j]):.3f}'
        )
    plt.savefig('tests/figs/outsample_ar_without_period.png', dpi=200)
    '''
    # VAR with period
    model_var = NtsVar(
        maxlags=maxlags,
        trend='n',
    ).fit(
        y=data.values[:train_size],
        X=dummy_vars.values[:train_size]
    )

    insample = model_var.insample(X=dummy_vars.values[:train_size])
    pred = model_var.predict(X=dummy_vars.values[train_size:])

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for VAR in-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[maxlags:train_size], insample[:, j], label='in-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[maxlags:train_size, j], insample[:, j]):.3f}, '
            f'mae: {mae(data.iloc[maxlags:train_size, j], insample[:, j]):.3f}'
        )
    plt.savefig('tests/figs/intsample_var_with_season.png', dpi=200)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for VAR out-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[train_size:], data.iloc[train_size:, j], label='real')
        ax[j].plot(data.index[train_size:], pred[:, j], label='out-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[train_size:, j], pred[:, j]):.3f}, '
            f'mae: {mae(data.iloc[train_size:, j], pred[:, j]):.3f}'
        )
    plt.savefig('tests/figs/outsample_var_with_season.png', dpi=200)
    '''
    # VAR without period
    model_var = NtsVar(
        maxlags=maxlags,
        trend='n',
    ).fit(
        y=data.values[:train_size],
        X=dummy_vars_no_period.values[:train_size]
    )

    insample = model_var.insample(X=dummy_vars_no_period.values[:train_size])
    pred = model_var.predict(X=dummy_vars_no_period.values[train_size:])

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for VAR in-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[maxlags:train_size], insample[:, j], label='in-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[maxlags:train_size, j], insample[:, j]):.3f}, '
            f'mae: {mae(data.iloc[maxlags:train_size, j], insample[:, j]):.3f}'
        )
    plt.savefig('tests/figs/intsample_var_without_period.png', dpi=200)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 8))
    fig.suptitle('Toy test for VAR out-sample predict:')
    for j in range(data.shape[1]):
        ax[j].plot(data.index[:train_size], data.iloc[:train_size, j], label='train')
        ax[j].plot(data.index[train_size:], data.iloc[train_size:, j], label='real')
        ax[j].plot(data.index[train_size:], pred[:, j], label='out-sample')
        ax[j].legend()
        ax[j].set_title(
            f'node-{data.columns[j]}, mape: {mape(data.iloc[train_size:, j], pred[:, j]):.3f}, '
            f'mae: {mae(data.iloc[train_size:, j], pred[:, j]):.3f}'
        )
    plt.savefig('tests/figs/outsample_var_without_period.png', dpi=200)
    '''