import logging

import xgboost as xgb
import pandas as pd

from networkts.base import BaseForecaster, Timeseries
from networkts.base import as_numpy_array


class NtsXgboost(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,
        nthread: int = 16,
        name: str = "XGB"
    ):
        self.nthread = nthread
        self.name = name
        super(BaseForecaster, self).__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        # y - array with traffic
        # X - array with temporal features
        self.model = xgb.XGBRegressor(nthread=self.nthread).fit(
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
