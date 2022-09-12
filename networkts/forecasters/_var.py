import logging

import numpy as np

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from statsmodels.tsa.vector_ar.var_model import VAR


class NtsVar(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
        self,        
        maxlags: int = 1,
        trend: str = 'n',
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
        try:
            self._model = VAR(
                            endog=as_numpy_array(y),
                            exog=as_numpy_array(X),
                            ).fit(
                                maxlags=self.maxlags,
                                trend=self.trend
                                )
        except:
            self.LOGGER.warning(f'VAR failed to fit with maxlags = {self.maxlags} '
                                f'so fall back to maxlags = 1')
            self._model = VAR(
                            endog=as_numpy_array(y),
                            exog=as_numpy_array(X),
                            ).fit(
                                maxlags=1,
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
                                    y=self._model.endog[-self.maxlags:],
                                    steps=n_timesteps,
                                    exog_future=X
                                    )
        else:
            y_pred = np.array([np.mean(self._y) for _ in range(n_timesteps)])
        return y_pred

    def insample(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        in_sample = self._model.forecast(
                                    y=self._model.endog[:self.maxlags],
                                    steps=n_timesteps - self.maxlags,
                                    exog_future=X[self.maxlags:]
                                    )
        return in_sample

    @classmethod
    def summary(self):
        return self._model.summary()

    @classmethod
    def get_model_coeff(self):
        return self._model.params[1:, :]