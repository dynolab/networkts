import os
import sys
import warnings

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import TransformedTargetRegressor

from networkts.utils import common
sys.path.append(common.CONF['directory']['path_networks'])
from forecasters.autoreg import NtsAutoreg
from forecasters.xgboost import NtsXgboost
from forecasters.holtwinter import NtsHoltWinter
from utils.sklearn_helpers import SklearnWrapperForForecaster, build_target_transformer
from cross_validation import ValidationBasedOnRollingForecastingOrigin as Valid
from decompositions.basic import log_target, inverse_log_target


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    df = pd.read_csv(
                os.path.join(
                    os.getcwd(),
                    common.CONF['datasets']['abilene']['root'],
                    common.CONF['datasets']['abilene']['edges_traffic']
                    ),
                index_col=0,     # abilene, totem
                # header=None,   # pemsd7  
                )
    
    '''
    df = pd.read_csv(
                os.path.join(os.getcwd(), 'data/PeMSD7/V_228.csv'),
                header=None
                )
    '''
    '''
    df = pd.read_csv(
        os.path.join(os.getcwd(), 'data/Totem/Link_counts.csv'),
        index_col=0
        )
    '''

    df = df.replace([0], 0.1)

    train_size = 1000
    test_size = 500
    period = 288
    delta_time = 5

    ind = np.array([el*delta_time for el in range(df.shape[0])])

    #f = open(f"valid_results/Abilene/window/score_ar_{train_size}.txt", "w")
    score_mape = []
    score_mae = []
    time = datetime.now()
    for i, feature in enumerate(df.columns.values[:2]):
        print(f'{i+1}/{len(df.columns.values)}')        
        cross_val = Valid(
                        n_test_timesteps=test_size,
                        n_training_timesteps=train_size,
                        n_splits=df.shape[0]//test_size - train_size//test_size,
                        max_train_size=np.Inf
                        )
        # Holt-winter
        
        model = build_target_transformer(
                        TransformedTargetRegressor,
                        SklearnWrapperForForecaster(NtsHoltWinter(
                                                    seasonal='additive',
                                                    seasonal_periods=period
                                                    )),
                        func=log_target,
                        inverse_func=inverse_log_target,
                        params=None,
                        inverse_params=None,
                        )
        

        # XGB
        '''
        model = build_target_transformer(
                                    TransformedTargetRegressor,
                                    SklearnWrapperForForecaster(NtsXgboost()),
                                    func=log_target,
                                    inverse_func=inverse_log_target,
                                    params=None,
                                    inverse_params=None,
                                    )
        '''

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
        
        t = cross_val.evaluate(
                        forecaster=model,
                        y=df[feature].values,
                        X=ind
                        )

        t = np.array(t)
        t1, t2 = t[:, 0], t[:, 1]
        score_mape += t1.tolist()
        score_mae += t2.tolist()
    
    time = datetime.now() - time

    score_mae = np.array(score_mae).reshape(-1)
    score_mape = np.array(score_mape).reshape(-1)

    print(np.mean(score_mape), np.median(score_mape))
    print(np.mean(score_mae), np.median(score_mae))

    '''
    for i in range(len(score_mae)):
        try:
            f.write(f'{score_mape[i]:.3f} {score_mae[i]:.3f}\n')
        except:
            f.write('None\n')

    try:
        f.write(f'\nAvg MAPE = {np.mean(score_mape):.3f}\n')
        f.write(f'MAPE median = {np.median(score_mape):.3f}\n')
        f.write(f'Avg MAE = {np.mean(score_mae):.3f}\n')
        f.write(f'MAE median = {np.median(score_mae):.3f}\n')
    except:
        f.write("\nNan score...\n")

    f.write(f'Time = {time.total_seconds()} sec')
    f.close()
    '''