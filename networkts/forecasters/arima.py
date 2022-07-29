import logging

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from statsmodels.tsa.arima.model import ARIMA


class NtsArima(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,
        order = (0, 0, 1),
        seasonal_order = (0, 0, 0, 0),
        name: str = 'ARIMA'
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = name
        super(BaseForecaster, self).__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        self._model = ARIMA(
                        endog=as_numpy_array(y),
                        exog=as_numpy_array(X, drop_first_column=True),
                        order=self.order,
                        seasonal_order=self.seasonal_order
                        ).fit()
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        y_pred = self._model.forecast(n_timesteps)
        return y_pred
