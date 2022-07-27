import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SSA(object):
    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(
        self,
        tseries,
        L: int = 50,
        noise_signal_split: int = 2,
        save_mem: bool = True
    ):
        """
        Decomposes the given time series with a singular-spectrum analysis.
        Assumes the values of the time series are recorded at equal intervals.

        Parameters
        ----------
        tseries :   The original time series, in the form of a Pandas Series,
                    NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the
            length of the time series.
        save_mem :  Conserve memory by not retaining the elementary matrices.
                    Recommended for long time series with thousands of values.
                    Defaults to True.

        Note: Even if an NumPy array or list is used for the initial time
        series, all time series returned will be in the form of a Pandas
        Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            er = 'Unsupported time series object. '
            er2 = 'Try Pandas Series, NumPy array or list.'
            raise TypeError(er+er2)

        # Checks to save us from ourselves
        self._N = len(tseries)
        if not 2 <= L <= self._N/2:
            er = "The window length must be in the interval [2, N/2]."
            raise ValueError(er)

        self._L = L
        self.noise_signal_split = noise_signal_split
        tseries = np.array(tseries).reshape(-1)
        self.orig_TS = pd.Series(tseries)
        self._K = self._N - self._L + 1

        # Embed the time series in a trajectory matrix
        self._X = np.array([self.orig_TS.values[i:L+i]
                            for i in range(0, self._K)]).T

        # Decompose the trajectory matrix
        self._U, self.Sigma, self._VT = np.linalg.svd(self._X)
        self._d = np.linalg.matrix_rank(self._X)

        self.TS_comps = np.zeros((self._N, self._d))

        self.save_mem = save_mem
        self._V = None
        self._X_elem = None

        self.Wcorr = None

    def save_memory(self):
        if not self.save_mem:
            # Construct and save all the elementary matrices
            self._X_elem = np.array([self.Sigma[i]*np.outer(self._U[:, i],
                                                            self._VT[i, :])
                                    for i in range(self._d)])

            # Diagonally average the elementary matrices, store them
            # as columns in array.
            for i in range(self._d):
                X_rev = self._X_elem[i, ::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean()
                                       for j in range(
                                                    -X_rev.shape[0]+1,
                                                    X_rev.shape[1])]

            self._V = self._VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self._d):
                X_elem = self.Sigma[i]*np.outer(self._U[:, i], self._VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean()
                                       for j in range(
                                                    -X_rev.shape[0]+1,
                                                    X_rev.shape[1])]
            er = "Re-run with save_mem=False to "
            er2 = "retain the elementary matrices."
            self._X_elem = er + er2

            # The V array may also be very large under these circumstances,
            # so we won't keep it.
            self._V = "Re-run with save_mem=False to retain the V matrix."

    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single
        Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self._d)
        else:
            n = self._d

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols,
                            index=self.orig_TS.index)

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the
        given indices. Returns a Pandas Series object with the reconstructed
        time series.

        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object,
        representing the elementary components to sum.
        """
        if isinstance(indices, int):
            indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        el1 = list(np.arange(self._L)+1)
        el2 = [self._L]*(self._K-self._L-1)
        el3 = list(np.arange(self._L)+1)[::-1]
        w = np.array(el1 + el2 + el3)

        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i])
                            for i in range(self._d)])
        F_wnorms = F_wnorms**-0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self._d)
        for i in range(self._d):
            for j in range(i+1, self._d):
                self.Wcorr[i, j] = abs(w_inner(
                                        self.TS_comps[:, i],
                                        self.TS_comps[:, j]
                                        ) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self._d

        if self.Wcorr is None:
            self.calc_wcorr()

        plt.figure(figsize=(15, 15))
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self._d:
            max_rnge = self._d-1
        else:
            max_rnge = max

        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)

    def ssa_target(self, y):
        y = np.array(y).reshape(-1)
        self.save_memory()
        self.calc_wcorr()
        res = np.array(abs(self.reconstruct(slice(0, self.noise_signal_split))))
        return res
    
    def inverse_ssa_target(self, y):
        # do nothing
        return y