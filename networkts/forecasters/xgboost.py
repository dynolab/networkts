import logging

from dataclasses import dataclass
from typing import Any
import xgboost as xgb
import numpy as np

from networkts.base import BaseForecaster, Timeseries
from networkts.base import as_numpy_array

@dataclass
class NtsXgboost(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)
    nthread: int = 1
    name: str = "XGB"
    random_state: int or np.random.RandomState = None
    learning_rate: float = 0.3
    max_depth: int = 6
    gamma: float = 0
    min_child_weight: float = 1
    max_delta_step: float = 0
    subsample: float = 1
    colsample_bytree: float = 1
    colsample_bylevel: float = 1
    colsample_bynode : float = 1
    reg_lambda: float = 1
    reg_alpha: float = 0
    num_parallel_tree: int = 1


    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        # y - array with traffic
        # X - array with temporal features
        self.model = xgb.XGBRegressor(
            nthread=self.nthread,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            num_parallel_tree=self.num_parallel_tree,
            ).fit(
                X=as_numpy_array(X),
                y=as_numpy_array(y),
                verbose=False
        )
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        y_pred = self.model.predict(as_numpy_array(X))
        return y_pred
