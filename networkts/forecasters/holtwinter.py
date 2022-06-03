from sttf.base import BaseForecaster
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NtsHoltWinter(BaseForecaster):
    def __init__(self, seasonal='additive', seasonal_periods=0):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super().__init__()

    def _fit(self, y, X):
        self.model = ExponentialSmoothing(
                                        endog=y,
                                        seasonal=self.seasonal,
                                        seasonal_periods=self.seasonal_periods,
                                        ).fit()
        return self

    def _predict(self, n_timesteps, X=None):
        y_pred = self.model.forecast(n_timesteps)
        return y_pred

    def _update(self, y, X):
        self.model = self.fit(self, y=y, X=X)
        return self
