import xgboost as xgb
import pandas as pd

from sttf.base import BaseForecaster
from sttf.utils.convert_time import time
from sttf.utils.create_features import create_features


class NtsXgboost(BaseForecaster):
    def __init__(self):
        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super(BaseForecaster, self).__init__()

    def _fit(self, y, X):
        # y - array with traffic
        # X - array with temporal features (count of mins)
        train_x = create_features([time(el) for el in X])
        self.model = xgb.XGBRegressor().fit(train_x, y, verbose=False)
        return self

    def _predict(self, n_timesteps, X=None):
        test_x = create_features([time(el) for el in X])
        y_pred = self.model.predict(test_x)
        return y_pred

    def _update(self, y, X):
        self.model = self.fit(self, y=y, X=X)
        return self
