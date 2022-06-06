import logging

from prophet import Prophet
import pandas as pd

from networkts.base import BaseForecaster, Timeseries, as_numpy_array, take_time_array
from networkts.utils.convert_time import time


class NtsProphet(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(self, yearly_seasonality=False, weekly_seasonality=False):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        super(BaseForecaster, self).__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        train = pd.DataFrame(columns=['ds', 'y'])
        train['ds'] = [time(el) for el in take_time_array(X)]
        train['y'] = as_numpy_array(y)
        self._model = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality
                        ).fit(train)
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        test = pd.DataFrame(
            [time(el) for el in take_time_array(X)],
            columns=['ds']
        )
        y_pred = self._model.predict(test)['yhat'].values
        return y_pred
