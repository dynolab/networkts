import os
import warnings
import pickle
import logging

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


LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}/{__file__}')
    
    nthread = cfg.multiprocessing.nthread
    controller = ThreadpoolController()
    @threadpool_limits.wrap(limits=nthread, user_api='blas')
    def forecast():
        dataset = call(cfg.dataset)    
        df = dataset.edge_timeseries.data
                
        test_size = 500
        period = dataset.period
        delta_time = dataset.delta_time
        df.index = np.array([el*delta_time for el in range(df.shape[0])])   # for pemsd7

        # setting the low limit for abilene's, totem's traffic
        if dataset.name in ['Abilene', 'Totem']:
            for feature in df.columns.values:
                df.loc[df[feature] < 1000, feature] = 1000

        decomposition = instantiate(cfg.decomposition)

        for train_size in [1000, 2000, 3000, 4000, 5000]:
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
                
                forecaster = instantiate(cfg.forecaster)
                    
                model = build_target_transformer(
                            TransformedTargetRegressor,
                            SklearnWrapperForForecaster(forecaster),
                            func = decomposition.transform,
                            inverse_func = decomposition.inverse_transform,
                            )

                t = cross_val.evaluate(
                                forecaster=model,
                                y=df[feature].values,
                                X=df.index.values
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
            with open(
                f'valid_results/{dataset.name}/window/'
                f'{decomposition.name}/'
                f'score_{forecaster.name}_{train_size}', 'wb'
                ) as file_pi:
                pickle.dump(score_dict, file_pi)
    forecast()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
    print('done')
