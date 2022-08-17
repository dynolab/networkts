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
from networkts.utils.create_features import create_features
from networkts.utils.convert_time import convert_time


LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}/{__file__}')
    
    nthread = cfg.multiprocessing.nthread
    controller = ThreadpoolController()
    @threadpool_limits.wrap(limits=nthread, user_api='blas')
    def forecast():
        dataset = call(cfg.dataset)    
        df = dataset.node_timeseries.data
        G = dataset.topology
                
        test_size = 500
        delta_time = dataset.delta_time
        df.index = np.array([el*delta_time for el in range(df.shape[0])])
        df.columns = df.columns.values.astype(str)

        # setting the low limit for abilene's, totem's traffic
        if dataset.name in ['Abilene', 'Totem']:
            for feature in df.columns.values:
                df.loc[df[feature] < 1000, feature] = 1000

        decomposition = instantiate(cfg.decomposition)

        dummy_vars = create_features([convert_time(_) for _ in df.index.values])

        for train_size in [1000, 2000, 3000, 4000, 5000]:
            LOGGER.info(f'Window size = {train_size}')
            score_mape = []
            score_mae = []
            time = datetime.now()
            for i, feature in enumerate(list(G.nodes)):
                LOGGER.info(f'{i+1}/{df.shape[1]}, {feature} node')
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
                'Time': time.total_seconds(),
                'Mape': score_mape,
                'Mae': score_mae
            }
            try:
                score_dict['Avg_mae'] = np.mean(score_mae)
                score_dict['Avg_mape'] = np.mean(score_mape)
                score_dict['Mae_median'] = np.median(score_mae)
                score_dict['Mape_median'] = np.median(score_mape)
            except:
                LOGGER.warning('Scores have None values')
                score_dict['Avg_mae'] = 'Value error: None score'
                score_dict['Avg_mape'] = 'Value error: None score'
                score_dict['Mae_median'] = 'Value error: None score'
                score_dict['Mape_median'] = 'Value error: None score'
            with open(
                f'valid_results/{dataset.name}/window/'
                f'{decomposition.name}/'
                f'nodes_score_{forecaster.name}_{train_size}', 'wb'
                ) as file_pi:
                pickle.dump(score_dict, file_pi)
    forecast()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
    print('done')
