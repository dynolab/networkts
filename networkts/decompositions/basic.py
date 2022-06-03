import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats

from networkts.decompositions.ssa import SSA


def exp_smoth(data, params=None):
    if params is None:
        a = 0.1
    else:
        a = params[0]
    s = ExponentialSmoothing(data).fit(smoothing_level=a, optimized=False)
    return s.fittedvalues


def moving_avg(data, params=None):
    if params is None:
        n = 100
    else:
        n = params[0]
    data = pd.DataFrame(data)
    data = data.rolling(n, min_periods=1).mean()
    return data.values


def scale_target(y, params=None):
    if params is None:
        shift = 0
        scale = 1
    else:
        shift = params[0]
        scale = params[1]
    return (y - shift) / scale


def inverse_scale_target(y, params=None):
    if params is None:
        shift = 0
        scale = 1
    else:
        shift = params[0]
        scale = params[1]
    return y * scale + shift


def log_target(y, params=None):
    return np.log(y)


def inverse_log_target(y, params=None):
    return np.exp(y)


def box_cox_transform(y, params=None):
    y = np.array(y).reshape(-1).tolist()
    if (len(y) < 1 or len(y) == y.count(y[0])):
        return np.log(y), 0
    return stats.boxcox(y)


def inverse_box_cox(y, lmbd, params=None):
    if lmbd == 0:
        return np.exp(y)
    return np.power(abs(y*lmbd+1), 1/lmbd)


def ssa_transform(y, params=None):
    if params is None:
        L = 50
        n = 2
    else:
        L = params[0]
        n = params[1]

    y = np.array(y).reshape(-1)
    y_ssa = SSA(tseries=y, L=L)
    y_ssa.save_memory()
    y_ssa.calc_wcorr()
    res = np.array(abs(y_ssa.reconstruct(slice(0, n))))
    return res


def nothing(y, params=None):
    return y


def fft_smoth(x, n_harm=10):
    # n_harm - number of harmonics in model
    n = x.size
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return abs(restored_sig + p[0] * t)
