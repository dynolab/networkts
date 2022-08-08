import logging

import numpy as np

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from statsmodels.tsa.ar_model import AutoReg


class NtsAutoreg(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,
        lags: int = 1,
        seasonal: bool = False,
        period: int = None,
        trend: str = 'n',
        cov_type: str = 'nonrobust',
        cov_kwds: dict = None,
        name: str = "AR",
    ):
        self.lags = lags
        self.seasonal = seasonal
        self.period = period
        self.trend = trend
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.name = name
        self.stable = True
        self._y = None
        super(BaseForecaster, self).__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        try:
            self._model = AutoReg(
                                endog=as_numpy_array(y),
                                exog=as_numpy_array(X),
                                lags=self.lags,
                                seasonal=self.seasonal,
                                period=self.period,
                                trend=self.trend,
                                ).fit(
                                    cov_type=self.cov_type,
                                    cov_kwds=self.cov_kwds
                                    )
        except:
            self.LOGGER.warning(f'AR failed to fit with lags = {self.lags} '
                                f'so fall back to lags = 1')
            self._model = AutoReg(
                                endog=as_numpy_array(y),
                                exog=as_numpy_array(X),
                                lags=1,
                                seasonal=self.seasonal,
                                period=self.period,
                                trend=self.trend,
                                ).fit(
                                    cov_type=self.cov_type,
                                    cov_kwds=self.cov_kwds
                                    )
        min_root = min(abs(self._model.roots))
        if min_root < 1:
            self.stable = False
            self._y = y
        else:
            self.stable = True
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        if self.stable:
            y_pred = self._model.forecast(
                                    steps=n_timesteps,
                                    exog=X
                                    )
        else:
            y_pred = np.array([np.mean(self._y) for _ in range(n_timesteps)])
        return y_pred
    
    def insample(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        in_sample = self._model.predict(
                                    start=0,
                                    end=n_timesteps-1,
                                    exog=X
                                    )
        return in_sample
    
    def summary(self):
        return self._model.summary()
