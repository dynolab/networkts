import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class LogTarget(object):
    def __init__(
        self,
        name: str = "Log"
    ):
        self.name = name
        super().__init__()

    def transform(self, y):
        return np.log(y)

    def inverse_transform(self, y):
        return np.exp(y)
    

class BoxcoxTarget(object):
    def __init__(
        self,
        name: str = "Boxcox"
    ):
        self.lmbd = None
        self.is_fitted: bool = False
        self.name = name
        super().__init__()

    def transform(self, y):
        self.is_fitted = True
        if len(y.shape) == 1:
            y = y.tolist()
            if (len(y) < 1 or len(y) == y.count(y[0])):
                res, self.lmbd = np.log(y), 0
                return res
            res, self.lmbd = stats.boxcox(y)
        else:
            res = np.empty(y.shape)
            self.lmbd = np.empty(y.shape[1])
            for j in range(y.shape[1]):
                if (len(y[:, j]) < 2 or len(y[:, j]) == list(y[:, j]).count(y[0, j])):
                    res[:, j], self.lmbd[j] = np.log(y[:, j]), 0
                    return res
                res[:, j], self.lmbd[j] = stats.boxcox(y[:, j])
        return res

    def inverse_transform(self, y):
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `target` first."
            )

        if isinstance(self.lmbd, int):
            if self.lmbd == 0:
                return np.exp(y)
            res = np.power(abs(y*self.lmbd+1), 1/self.lmbd)

        else:
            res = np.empty(y.shape)
            for j in range(y.shape[1]):
                if self.lmbd[j] == 0:
                    res[:, j] = np.exp(y[:, j])
                else:
                    res[:, j] = np.power(abs(y[:, j]*self.lmbd[j]+1), 1/self.lmbd[j])
        return res


class ESTarget(object):
    def __init__(
        self,
        smoothing_level: float = 0.1,
        name: str = 'Exp_smooth'
    ):
        self.smoothing_level = smoothing_level
        self.name = name
        super().__init__()

    def transform(self, y):
        if len(y.shape) < 2:
            s = ExponentialSmoothing(y).fit(
                                        smoothing_level=self.smoothing_level,
                                        optimized=False
                                        )
            res = s.fittedvalues
        else:
            res = np.empty(y.shape)
            for j in range(y.shape[1]):
                s = ExponentialSmoothing(y[:, j]).fit(
                                        smoothing_level=self.smoothing_level,
                                        optimized=False
                                        )
                res[:, j] = s.fittedvalues
        return res

    def inverse_transform(self, y):
        # do nothing
        return y


class MATarget(object):
    def __init__(
        self,
        num_previous_points: int = 100,
        name: str = 'Moving_avg'
    ):
        self.num_previous_points = num_previous_points
        self.name = name
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