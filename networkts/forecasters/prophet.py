from prophet import Prophet
import pandas as pd

from sttf.base import BaseForecaster
from sttf.utils.convert_time import time


class NtsProphet(BaseForecaster):
    def __init__(self, yearly_seasonality=False, weekly_seasonality=False):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality

        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super(BaseForecaster, self).__init__()

    def _fit(self, y, X):
        train = pd.DataFrame(columns=['ds', 'y'])
        train['ds'] = [time(el) for el in X]
        train['y'] = y
        self.model = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality
                        ).fit(train)
        return self

    def _predict(self, n_timesteps, X=None):
        test = pd.DataFrame([time(el) for el in X], columns=['ds'])
        y_pred = self.model.predict(test)['yhat'].values
        return y_pred

    def _update(self, y, X):
        self.model = self.fit(self, y=y, X=X)
        return self
