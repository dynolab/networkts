import os
import sys
import warnings
import pickle

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

    df = df.replace([0], 0.1)

    train_size = 1000
    test_size = 500
    period = 288
    delta_time = 5

    ind = np.array([el*delta_time for el in range(df.shape[0])])

    score_mape = []
    score_mae = []
    time = datetime.now()
    for i, feature in enumerate(df.columns.values):
        print(f'{i+1}/{len(df.columns.values)}')        
        cross_val = Valid(
                        n_test_timesteps=test_size,
                        n_training_timesteps=train_size,
                        n_splits=df.shape[0]//test_size - train_size//test_size,
                        max_train_size=np.Inf
                        )
        # Holt-winter
        '''
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
        '''

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

    score_file_name = 'your file name'
    score_dict = {
        'Avg_mape': np.mean(score_mape),
        'Mape_median': np.median(score_mape),
        'Avg_mae': np.mean(score_mae),
        'Mae_median': np.median(score_mae),
        'Time': time.total_seconds(),
        'Mape': score_mape,
        'Mae': score_mae
        }
    with open(score_file_name, 'wb') as file_pi:
        pickle.dump(score_dict, file_pi)