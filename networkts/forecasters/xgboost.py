import logging

import xgboost as xgb
import pandas as pd

from networkts.base import BaseForecaster, Timeseries, as_numpy_array, take_time_array
from networkts.utils.convert_time import time
from networkts.utils.create_features import create_features


class NtsXgboost(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(self):
        super(BaseForecaster, self).__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        # y - array with traffic
        # X - array with temporal features (count of mins)
        train_x = create_features([time(el) for el in take_time_array(X)])
        self.model = xgb.XGBRegressor().fit(
            train_x,
            as_numpy_array(y),
            verbose=False
        )
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        test_x = create_features([time(el) for el in take_time_array(X)])
        y_pred = self.model.predict(test_x)
        return y_pred
