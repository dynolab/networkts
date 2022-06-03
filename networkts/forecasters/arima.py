from networkts.base import BaseForecaster
from statsmodels.tsa.arima.model import ARIMA


class NtsArima(BaseForecaster):
    def __init__(self, order=(0, 0, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order

        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super(BaseForecaster, self).__init__()

    def _fit(self, y, X):
        self.model = ARIMA(
                        endog=y,
                        exog=X,
                        order=self.order,
                        seasonal_order=self.seasonal_order
                        ).fit()
        return self

    def _predict(self, n_timesteps, X=None):
        y_pred = self.model.forecast(n_timesteps)
        return y_pred

    def _update(self, y, X):
        self.model = self.fit(self, y=y, X=X)
        return self
