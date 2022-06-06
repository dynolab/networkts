import logging

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from statsmodels.tsa.ar_model import AutoReg


class NtsAutoreg(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,
        lags=1,
        seasonal=False,
        period=None
    ):
        self.lags = lags
        self.seasonal = seasonal
        self.period = period
        super(BaseForecaster, self).__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        try:
            self._model = AutoReg(
                                endog=as_numpy_array(y),
                                exog=as_numpy_array(X, drop_first_column=True),
                                lags=self.lags,
                                seasonal=self.seasonal,
                                period=self.period,
                                ).fit()
        except:
            self.LOGGER.warning(f'AR failed to fit with lags = {self.lags} '
                                f'so fall back to lags = 1')
            self._model = AutoReg(
                                endog=as_numpy_array(y),
                                exog=as_numpy_array(X, drop_first_column=True),
                                lags=1,
                                seasonal=self.seasonal,
                                period=self.period,
                                ).fit()
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        y_pred = self._model.forecast(n_timesteps)
        return y_pred
