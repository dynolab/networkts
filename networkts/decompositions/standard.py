import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class LogTarget(object):
    def __init__(self):
        super().__init__()

    def transform(self, y):
        return np.log(y)

    def inverse_transform(self, y):
        return np.exp(y)
    

class BoxcoxTarget(object):
    def __init__(self):
        self.lmbd = None
        self.is_fitted: bool = False
        super().__init__()

    def transform(self, y):
        y = np.array(y).reshape(-1).tolist()
        self.is_fitted = True
        if (len(y) < 1 or len(y) == y.count(y[0])):
            res, self.lmbd = np.log(y), 0
            return res
        res, self.lmbd = stats.boxcox(y)
        return res

    def inverse_transform(self, y):
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `target` first."
            )
        if self.lmbd == 0:
            return np.exp(y)
        return np.power(abs(y*self.lmbd+1), 1/self.lmbd)


class ESTarget(object):
    def __init__(
        self,
        smoothing_level: float = 0.1
    ):
        self.smoothing_level = smoothing_level
        super().__init__()

    def transform(self, y):
        s = ExponentialSmoothing(y).fit(
                                    smoothing_level=self.smoothing_level,
                                    optimized=False
                                    )
        return s.fittedvalues

    def inverse_transform(self, y):
        # do nothing
        return y


class MATarget(object):
    def __init__(
        self,
        num_previous_points: int = 100
    ):
        self.num_previous_points = num_previous_points
        super().__init__()

    def transform(self, y):
        data = pd.DataFrame(y)
        data = data.rolling(
                    self.num_previous_points,
                    min_periods=1
                    ).mean()
        return data.values

    def inverse_transform(self, y):
        # do nothing
        return y

class NotFittedError(Exception):
    pass