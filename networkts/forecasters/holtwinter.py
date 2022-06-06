import logging

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NtsHoltWinter(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(self, seasonal='additive', seasonal_periods=0):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        super().__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        self._model = ExponentialSmoothing(
                                        endog=as_numpy_array(y),
                                        seasonal=self.seasonal,
                                        seasonal_periods=self.seasonal_periods,
                                        ).fit()
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        y_pred = self._model.forecast(n_timesteps)
        return y_pred
