import logging

import lightgbm as lgb

from networkts.base import BaseForecaster, Timeseries
from networkts.base import as_numpy_array


class NtsLightgbm(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,
        num_round: int = 1000,
        num_leaves: int = 32,
        num_threads: int = 16,
        learning_rate: float = 0.1,
        name: str = 'LightGBM'
    ):
        self._is_fitted = False
        self.num_round = num_round
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.num_threads = num_threads
        self.name = name
        self.params = {
            "objective": "regression",
            "metric": ["mape", "mae"],
            "num_leaves": self.num_leaves,
            "num_threads": self.num_threads,
            "learning_rate": self.learning_rate,
            'verbose': -1,
            }
        super().__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        # y - array with traffic
        # X - array with temporal features
        train_data = lgb.Dataset(as_numpy_array(X), label=as_numpy_array(y))
        self.model = lgb.train(self.params, train_data, self.num_round)
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        y_pred = self.model.predict(as_numpy_array(X))
        return y_pred
