import logging

import numpy as np

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from statsmodels.tsa.vector_ar.var_model import VAR


class NtsVar(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,        
        maxlags: int = 1,
        trend: str = 'c',
        ic: str = None,
        name: str = 'VAR'
    ):       
        self.maxlags = maxlags
        self.trend = trend
        self.ic = ic
        self.name = name
        self.stable = True
        self._y = None
        super().__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        self._y = None
        self._model = VAR(
                        endog=as_numpy_array(y),
                        exog=as_numpy_array(X),
                        ).fit(
                            maxlags=self.maxlags,
                            trend=self.trend
                            )
        n = self._model.roots.shape[0]//y.shape[1]
        min_root = min(abs(self._model.roots[:n]))
        if min_root < 1:
            self.stable = False
            self._y = y[:, 0]
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
                                    y=self._model.endog,
                                    steps=n_timesteps,
                                    exog_future=X[-n_timesteps:]
                                    )[:, 0]
        else:
            y_pred = np.array([np.mean(self._y) for _ in range(n_timesteps)])
        return y_pred