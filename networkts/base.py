from typing import Any, Union, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as _BaseEstimator


Timeseries = Union[pd.Series, pd.DataFrame, np.ndarray] 


# COPY FROM sktime.base._base.py
class BaseEstimator(_BaseEstimator):
    """
    Base class for defining estimators in sktime.
    Extends sktime's BaseObject to include basic functionality for
    fittable estimators.
    """

    def __init__(self):
        self._is_fitted = False
        super(BaseEstimator, self).__init__()

    @property
    def is_fitted(self):
        """Whether `fit` has been called."""
        return self._is_fitted

    def check_is_fitted(self):
        """Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )


# HEAVILY MODIFIED COPY FROM sktime.forecasting.base._base.py
class BaseForecaster(BaseEstimator):
    """Base forecaster template class.

    The base forecaster specifies the methods and method
    signatures that all forecasters have to implement.

    Specific implementations of these methods is deferred to concrete
    forecasters.
    """

    def __init__(self):
        self._is_fitted = False

        self._X = None
        self._y = None
        self._model = None

        super(BaseForecaster, self).__init__()

    def fit(
        self,
        X: Timeseries,
        y: Timeseries,
        **kwargs,
    ):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.

        Returns
        -------
        self : Reference to self.
        """
        # If fit is called, fitted state is re-set
        self._is_fitted = False

        # Pass to inner fit
        if X is None:
            X = np.arange(len(y))
        self._fit(X=X, y=y, **kwargs)

        # this should happen last
        self._X = X
        self._y = y
        self._is_fitted = True

        return self

    def predict(
        self,
        X: Timeseries,
        **kwargs,
    ):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
            Many forecasting methods imply that this exogeneous time 
            series is a continuation of the one used during the 
            training.

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecasted endogeneous time series.
        """
        # handle inputs

        self.check_is_fitted()

        # this is how it is supposed to be after the refactor is complete
        # and effective
        y_pred = self._predict(X=X, **kwargs)
        return y_pred

    def fit_predict(
        self,
        X: Timeseries,
        y: Timeseries,
        fit_kwargs={},
        predict_kwargs={},
    ):
        """Fit and forecast time series at future horizon.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
            It must be longer than y.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecasted endogeneous time series
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        # apply fit and then predict
        n_timesteps = len(y)
        self._fit(X=X[:n_timesteps], y=y, **fit_kwargs)
        self._is_fitted = True
        # call the public predict to avoid duplicating output conversions
        #  input conversions are skipped since we are using X_inner
        return self.predict(X=X[n_timesteps:], **predict_kwargs)

    def update(
        self,
        X: Timeseries,
        y: Timeseries,
        update_params=True,
    ):
        """Update fitted parameters to fit new data.

        If no estimator-specific update method has been implemented,
        default fall-back is as follows:
            update_params=True: fitting to all observed data so far
            update_params=False: remembers data only with no update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        self.check_is_fitted()

        # checks and conversions complete, pass to inner fit
        if X is None:
            X = np.arange(len(y))
        self._update(X=X, y=y, update_params=update_params)

        return self

    def update_predict_single(
        self,
        X: Timeseries,
        y: Timeseries,
        mode=None,
    ):
        """Update model with new data and make forecasts.

        This method is useful for updating and making forecasts in a single
        step.

        If no estimator-specific update method has been implemented,
        default fall-back is first update, then predict.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
            It must be longer than y.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecasted endogeneous time series
        """
        self.check_is_fitted()
        return self._update_predict_single(
            X=X,
            y=y,
            mode=mode,
        )

    def predict_residuals(
        self,
        X: Timeseries,
        y: Optional[Timeseries] = None,
        **kwargs,
    ):
        """Return residuals of time series forecasts.

        If y is None, in-sample forecast is done and 
        then the residual is calculated.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if None, the y seen so far (self._y) are used, in particular:
                if preceded by a single fit call, then in-sample residuals are
                produced

        Returns
        -------
        y_res : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecast residuals
        """
        # if no y is passed, the so far observed y is used
        if y is None:
            y = self._y
            X = self._X

        y_pred = self.predict(X=X, **kwargs)
        y_res = y - y_pred

        return y_res

    def score(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        """Scores forecast against ground truth, using MAPE.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to score.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to score.

        Returns
        -------
        score : float
            MAPE loss of self.predict(X) with respect to y.
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        # symmetric=True is default for mean_absolute_percentage_error
        from sklearn.metrics import (
            mean_absolute_percentage_error,
        )

        if n_timesteps is None:
            n_timesteps = y.shape[0]
        return mean_absolute_percentage_error(y, self.predict(X))

    def get_fitted_params(self):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        raise NotImplementedError("abstract method")

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        """Fit forecaster to training data.

            core logic

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.

        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def _predict(
        self,
        X: Timeseries,
    ):
        """Forecast time series at future horizon.

            core logic

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to predict from.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecasted endogeneous time series
        """
        raise NotImplementedError("abstract method")

    def _update(
        self,
        X: Timeseries,
        y: Timeseries,
        mode: Optional[Any] = None,
    ):
        """Update time series to incremental training data.

        Writes to self:
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or 2D np.array
            Exogeneous time series to fit to.
            Must include time axis. By default, the first column
            in DataFrame or np.ndarray is considered as a time axis.
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
        mode : str, optional (default=None)
            Particular update mode specified by a concrete class.

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecasted endogeneous time series
        """
        # default to re-fitting if update is not implemented
        print(
            f"NotImplementedWarning: {self.__class__.__name__} "
            f"does not have a custom `update` method implemented. "
            f"{self.__class__.__name__} will be refit each time "
            f"`update` is called."
        )
        # refit with updated data, not only passed data
        self.fit(
            X=concat_timeseries(self._X, X),
            y=concat_timeseries(self._y, y)
        )
        # todo: should probably be self._fit, not self.fit
        # but looping to self.fit for now to avoid interface break

        return self

    def _update_predict_single(
        self,
        X: Timeseries,
        y: Timeseries,
        mode: Optional[Any] = None,
    ):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self.update(X, y, mode=mode)
        return self.predict(X)


def take_time_array(
    x: Timeseries,
):
    if isinstance(x, pd.DataFrame):
        return x[x.columns[0]]
    elif isinstance(x, pd.Series):
        return x
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f'Unknown type of object: {type(x)}')


def as_numpy_array(
    x: Timeseries,
    drop_first_column=False,
):
    if isinstance(x, pd.DataFrame):
        if drop_first_column:
            return None if x.shape[1] == 1 else x.values[:,1:]
        else:
            return x.values
    elif isinstance(x, pd.Series):
        return x.values
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f'Unknown type of object: {type(x)}')


def concat_timeseries(
    ts_1: Timeseries,
    ts_2: Timeseries,
):
    assert type(ts_1) == type(ts_2)
    if isinstance(ts_1, pd.DataFrame) or isinstance(ts_1, pd.Series):
        return pd.concat((ts_1, ts_2))
    elif isinstance(ts_1, np.ndarray):
        return np.concatenate((ts_1, ts_2))
    else:
        raise ValueError(f'Unknown time series type: {type(ts_1)}')


class NotFittedError(Exception):
    pass
