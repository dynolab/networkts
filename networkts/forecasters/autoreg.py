from networkts.base import BaseForecaster
from statsmodels.tsa.ar_model import AutoReg


class NtsAutoreg(BaseForecaster):
    def __init__(self, lags=1, seasonal=False, period=None):
        self.lags = lags
        self.seasonal = seasonal
        self.period = period

        self.model = None
        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super(BaseForecaster, self).__init__()

    def _fit(self, y, X):
        try:
            self.model = AutoReg(
                                endog=y,
                                exog=X,
                                lags=self.lags,
                                seasonal=self.seasonal,
                                period=self.period,
                                ).fit()
        except:
            self.model = AutoReg(
                                endog=y,
                                exog=X,
                                lags=1,
                                seasonal=self.seasonal,
                                period=self.period,
                                ).fit()
        return self

    def _predict(self, n_timesteps, X=None):
        y_pred = self.model.forecast(n_timesteps)
        return y_pred

    def _update(self, y, X):
        self.model = self.fit(self, y=y, X=X)
        return self
