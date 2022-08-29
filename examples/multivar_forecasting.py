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
from networkts.utils.create_dummy_vars import *
from networkts.decompositions.standard import *


LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}')

    nthread = cfg.multiprocessing.nthread
    controller = ThreadpoolController()
    @threadpool_limits.wrap(limits=nthread, user_api='blas')
    def forecast():
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

        # dummy variables for one hop VAR
        dummy_vars = create_dummy_vars(df.index, period)

        # smoothing
        decomposition = instantiate(cfg.decomposition)
        if 'log' not in decomposition.name:
            data = decomposition.transform(np.log(df.values))
            df = pd.DataFrame(np.exp(data), columns=df.columns, index=df.index)

        for train_size in [_*1000 for _ in range(1, 6)]:
            LOGGER.info(f'Window size = {train_size}')
            score_mape = []
            score_mae = []
            time = datetime.now()
            for i, feature in enumerate(list(G.nodes)):
                neighbors = list(G.neighbors(feature))
                if len(neighbors):
                    LOGGER.info(f'{i+1}/{df.shape[1]}, {feature} node')
                    cols = np.array([feature] + neighbors).astype(str)      # VAR
                    #cols = np.array(neighbors).astype(str)                 # XGB
                    data = df[cols]
                    #target = df[feature]   # XGB

                    # dummy variables for one hop XGB
                    #dummy_vars = create_xgb_dummy_vars(data, train_size)

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
                                func = LogTarget().transform,
                                inverse_func = LogTarget().inverse_transform,
                                )
                    
                    t = cross_val.evaluate(
                                    forecaster=model,
                                    y=data.values,          # VAR
                                    #y=target.values,       # XGB
                                    X=dummy_vars.values
                                    )

                    t = np.array(t)
                    t1, t2 = t[:, 0], t[:, 1]
                    score_mape += t1.tolist()
                    score_mae += t2.tolist()
                else:
                    LOGGER.warning(f"Node {feature} doesn't have neighbors")
                    continue
            
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
            
            directory = f'valid_results/{dataset.name}/window/' + \
                        f'{decomposition.name}+log/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(
                os.path.join(directory,
                            f'score_{forecaster.name}_{train_size}'),
                'wb'
                ) as file_pi:
                pickle.dump(score_dict, file_pi)
    
    forecast()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
    print('done')