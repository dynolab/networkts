import os
import warnings
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from sklearn.compose import TransformedTargetRegressor
from threadpoolctl import ThreadpoolController, threadpool_limits

import networkts
from networkts.utils import common
from networkts.utils.sklearn_helpers import SklearnWrapperForForecaster, build_target_transformer
from networkts.cross_validation import ValidationBasedOnRollingForecastingOrigin as Valid
from networkts.decompositions.basic import *
from networkts.forecasters._var import NtsVar
from networkts.utils.create_dummy_vars import create_dummy_vars


common.set_config('config/config.yaml')
nthread = common.CONF['multiprocessing']['nthread']
controller = ThreadpoolController()
@threadpool_limits.wrap(limits=nthread, user_api='blas')
def func():
    warnings.filterwarnings('ignore')

    df = pd.read_csv(
                os.path.join(
                    os.getcwd(),
                    common.CONF['datasets']['pemsd7']['root'],
                    common.CONF['datasets']['pemsd7']['speeds_file']
                    ),
                # index_col=0,      # abilene, totem
                header=None,        # pemsd7
                )
            
    test_size = 500
    period = 288
    delta_time = 5

    df.index = np.array([el*delta_time for el in range(df.shape[0])])   # for pemsd7

    '''
    for feature in df.columns.values:
        df.loc[df[feature] < 1000, feature] = 1000
    '''

    G = nx.read_adjlist(
                    os.path.join(
                            common.CONF['datasets']['pemsd7']['root'],
                            common.CONF['datasets']['pemsd7']['topology_adjlist_file']
                            ),
                    create_using=nx.DiGraph)

    # SSA
    L = 50
    n = 2

    # dummy variables for one hop forecasting
    dummy_vars = create_dummy_vars(df.index, period)

    for train_size in [1000, 2000, 3000, 4000, 5000]:
        score_mape = []
        score_mae = []
        time = datetime.now()
        for i, feature in enumerate(list(G.nodes)):
            print(f'{i+1}/{len(df.columns.values)}')
            cols = np.array([feature] + list(G.neighbors(feature))).astype(int)
            data = df[cols]

            cross_val = Valid(
                            n_test_timesteps=test_size,
                            n_training_timesteps=train_size,
                            n_splits=df.shape[0]//test_size - train_size//test_size,
                            max_train_size=np.Inf
                            )

            # XGB
            '''
            model = build_target_transformer(
                                        TransformedTargetRegressor,
                                        SklearnWrapperForForecaster(NtsXgboost(nthread=nthread)),
                                        func=log_target,
                                        inverse_func=inverse_log_target,
                                        params=None,
                                        inverse_params=None,
                                        )
            '''

            # VAR
            
            model = build_target_transformer(
                        TransformedTargetRegressor,
                        SklearnWrapperForForecaster(
                            NtsVar(
                                maxlags=300,
                                trend='ctt',
                                )),
                        func=log_target,
                        inverse_func=inverse_log_target,
                        params=None,
                        inverse_params=None,
                        )
            
            '''
            # Lgb
            model = build_target_transformer(
                        TransformedTargetRegressor,
                        SklearnWrapperForForecaster(
                            NtsLightgbm(
                                num_round=1000,
                                num_leaves=32,
                                num_threads=nthread,
                                learning_rate=0.1,
                                )),
                        func=log_target,
                        inverse_func=inverse_log_target,
                        params=None,
                        inverse_params=None,
                        )
            '''
            t = cross_val.evaluate(
                            forecaster=model,
                            y=data.values,
                            X=dummy_vars.values
                            )

            t = np.array(t)
            t1, t2 = t[:, 0], t[:, 1]
            score_mape += t1.tolist()
            score_mae += t2.tolist()
        
        time = datetime.now() - time

        score_mae = np.array(score_mae).reshape(-1)
        score_mape = np.array(score_mape).reshape(-1)

        score_dict = {
            'Avg_mape': np.mean(score_mape),
            'Mape_median': np.median(score_mape),
            'Avg_mae': np.mean(score_mae),
            'Mae_median': np.median(score_mae),
            'Time': time.total_seconds(),
            'Mape': score_mape,
            'Mae': score_mae
        }
        with open(f'valid_results/PeMSD7/window/log/score_var_{train_size}', 'wb') as file_pi:
            pickle.dump(score_dict, file_pi)
    
func()
print('done')