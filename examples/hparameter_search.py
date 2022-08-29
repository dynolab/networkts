import os
import warnings
import pickle
import logging
import random

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import TransformedTargetRegressor
from threadpoolctl import ThreadpoolController, threadpool_limits

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call

import networkts
from networkts.utils.sklearn_helpers import SklearnWrapperForForecaster
from networkts.utils.sklearn_helpers import build_target_transformer
from networkts.cross_validation import ValidationBasedOnRollingForecastingOrigin as Valid
from networkts.utils.create_dummy_vars import *
from networkts.decompositions.standard import *


LOGGER = logging.getLogger(__name__)
print = LOGGER.info


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}')

    nthread = cfg.multiprocessing.nthread
    controller = ThreadpoolController()
    @threadpool_limits.wrap(limits=nthread, user_api='blas')
    def param_search():
        dataset = call(cfg.dataset)    
        df = dataset.node_timeseries.data
        df.columns = df.columns.values.astype(str)
                
        test_size = 500
        period = dataset.period
        delta_time = dataset.delta_time
        
        df.set_index(np.array([el*delta_time for el in range(df.shape[0])]), inplace=True)

        # setting the low limit for abilene's, totem's traffic
        if dataset.name in ['Abilene', 'Totem']:
            for feature in df.columns.values:
                df.loc[df[feature] < 1000, feature] = 1000

        G = dataset.topology

        # smoothing
        decomposition = instantiate(cfg.decomposition)
        if 'log' not in decomposition.name:
            data = decomposition.transform(np.log(df.values))
            df = pd.DataFrame(np.exp(data), columns=df.columns, index=df.index)

        window_size = 3000
        feature = list(G.nodes)[random.randint(0, G.number_of_nodes())]
        while not len(feature):
            feature = list(G.nodes)[random.randint(0, G.number_of_nodes())]

        neighbors = list(G.neighbors(feature))
        
        cols = np.array(neighbors).astype(str)
        data = df[cols]
        target = df[feature]

        # dummy variables for one hop XGB
        dummy_vars = create_xgb_dummy_vars(data, window_size)

        cross_val = Valid(
                    n_test_timesteps=test_size,
                    n_training_timesteps=window_size,
                    n_splits=df.shape[0]//test_size - window_size//test_size,
                    max_train_size=np.Inf
                    )
        
        forecaster = instantiate(cfg.forecaster)
        model = build_target_transformer(
                                TransformedTargetRegressor,
                                SklearnWrapperForForecaster(forecaster),
                                func = LogTarget().transform,
                                inverse_func = LogTarget().inverse_transform,
                                )
                
        hparams = {
            'learning_rate': [0.01, 0.5],
            'gamma': [0, 5],
            'max_depth': [0, 10],
            'min_child_weight': [0, 120],
            'max_delta_step': [0, 10],
            'subsample': [0.5, 1],
            'colsample_bytree': [0.5, 1],
            'colsample_bylevel': [0.1, 1],
            'colsample_bynode' : [0.1, 1],
            'reg_lambda': [0, 1000],
            'reg_alpha': [0, 1000],
            'num_parallel_tree': [1, 5],
        }

        gs = cross_val.grid_search(
                                forecaster=model,
                                param_grid=hparams,
                                y=target.values,
                                X=dummy_vars.values,
                                verbose=3,
                                )

        directory = f'hparameters_search/{dataset.name}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(
            os.path.join(directory,
                        f'hparameters_{forecaster.name}_{window_size}'),
            'wb'
            ) as file_pi:
            pickle.dump(gs, file_pi)
        
        name = os.path.join(
                        directory,
                        f'hparameters_{forecaster.name}_{window_size}'
        )
        with open(name, 'rb') as f:
            gs2 = pickle.load(f)
        print(f'Best params:\n{gs2.best_params_};'
            '\nBest score:\n{abs(gs2.best_score_)}')
    
    param_search()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
    print('done')