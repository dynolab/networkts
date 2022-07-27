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
        ic: str = None
    ):       
        self.maxlags = maxlags
        self.trend = trend
        self.ic = ic
        self._y = None
        super().__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        self._model = VAR(
                        endog=as_numpy_array(y),
                        exog=as_numpy_array(X),
                        ).fit(
                            maxlags=self.maxlags,
                            trend=self.trend
                            )
        #print(self._model.roots.astype('float'))
        #if min(abs(self._model.roots[0])) < 1:
        #    self._y = y[0]
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        if self._y is not None and False:
            y_pred = np.array([np.mean(self._y) for _ in range(n_timesteps)])
        else:
            y_pred = self._model.forecast(
                                    y=self._model.endog,
                                    steps=n_timesteps,
                                    exog_future=X[-n_timesteps:]
                                    )[:, 0]
        return y_pred