# TODO: COPY FROM mlresn!

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import make_scorer

from sklearn.base import BaseEstimator


class ValidationBasedOnRollingForecastingOrigin:
    """Validation tools based on rolling forecasting origin.
    It implies that we are given a long time series which is
    to be split in training/validation pieces, and the prediction
    is performed from the end of the training piece.
    As we assume a really long time series, it can be separated
    into `n_splits` pairs of training/validation pieces.
    There can be two modes of splitting.
    The first mode takes the original time series and creates
    new splits by moving the end point of the training piece further
    and further with a fixed validation piece size
    defined by `n_test_timesteps`.
    To activate this mode, you must set `n_training_timesteps = None`.
    Here is an example for `n_test_timesteps = 3` where `][` denotes
    the forecasting origin:
    ```plain
    Original time series:
    |------------------------------------|

    Split #1:
    |-][---|
     T   V

    Split #2:
    |--][---|
     T    V
    Split #3:
    |---][---|
      T    V

    Split #4:
    |----][---|
      T     V

    ...
    ```

    The second mode fixes both the maximum training piece size controlled by
    `n_training_timesteps` abd the validation piece size.
    Here is an example for `n_training_timesteps = 2` and
    `n_test_timesteps = 3`:
    ```plain
    Original time series:
    |------------------------------------|

    Split #1:
    |-][---|
     T   V

    Split #2:
    |--][---|
     T    V
    Split #3:
     |--][---|
      T    V

    Split #4:
      |--][---|
       T    V

    ...
    ```

    See https://otexts.com/fpp3/tscv.html for additional discussion.

    Parameters
    ----------
    metric : function with the signature compatible with Regression metrics
        from sklearn.metrics. The value of this metric is used for
        model evaluation.
    n_training_timesteps : int
        Maximum number of time steps of the piece of time
        series used for training
    n_test_timesteps : int
        Number of time steps of the piece of time series used
        for validation of the trained model
    n_splits : int
        Required number of training/validation piece pairs
    """
    def __init__(
            self,
            metric=[mape, mae],
            n_training_timesteps=None,
            n_test_timesteps=10,
            n_splits=10,
            max_train_size=10000
            ):
        self.metric = metric
        self.n_training_timesteps = n_training_timesteps
        self.n_test_timesteps = n_test_timesteps
        self.n_splits = n_splits
        self.max_train_size = max_train_size

    def evaluate(self, forecaster, data, dummy_vars, original_data, log):
        """Evaluate the performance of the given forecaster using
        the time series passed via `y` and splitting it into
        training/validation pieces as specified by the constructor.

        Parameters
        ----------
        data:           array-like, shape (n_timesteps x n_endo)
                        Time series which will be split into training/validation pieces.
        dummy_vars:     array-like or function, shape (n_timesteps x n_exo), optional
                        (default=None). Exogeneous time series to adjust to. Can also
                        be understood as a control signal or some externally available
                        information.
        original_data:  array-like, shape (n_timesteps x n_endo)
                        Time series will be used to calculate score metrics.
        log:            boolean
                        Requirement of inverse logarithm transform.

        Returns
        -------
        A list of metric values for each split
        """
        return [self.score(y_true, y_pred, original_data[test_ind], log)
                for test_ind, y_pred, y_true in self.prediction_generator(
                                                forecaster,
                                                data,
                                                dummy_vars
                                                )]

    def prediction_generator(self, forecaster, y, X):
        """Generator yielding a triple `(test_index, y_pred, y_test)`
        for each pair of training/validation pieces where `test_index`
        is an array of indices corresponding to a validation piece in the
        original time series `y` and `y_pred` and `y_test` correspond to
        a model forecast and a true validation piece of time series
        respectively.

        Parameters
        ----------
        forecaster : BaseForecaster
            Forecaster model to be evaluated
        y : array-like, shape (n_timesteps x n_endo)
            Time series which will be split into training/validation pieces.
        X : array-like or function, shape (n_timesteps x n_exo), optional
            (default=None). Exogeneous time series to adjust to. Can also
            be understood as a control signal or some externally available
            information.

        Yields
        -------
        test_index : array-like, shape (n_timesteps,)
            Indices corresponding to the next validation piece
            in the original time series
        y_pred : array-like, shape (n_timesteps x n_endo)
            Model forecast for the next validation piece
        y[test_index] : array-like, shape (n_timesteps x n_endo)
            True time series for the next validation piece
        """

        ts_cv = TimeSeriesSplit(n_splits=self.n_splits,
                                max_train_size=self.n_training_timesteps,
                                test_size=self.n_test_timesteps)

        for train_index, test_index in ts_cv.split(y):
            # limit for train_size
            if train_index[-1] > self.max_train_size:
                break
            
            forecaster.fit(y=y[train_index], X=X[train_index])
            y_pred = forecaster.predict(X=X[test_index])
            yield test_index, y_pred, y[test_index]

    def grid_search(
        self,
        forecaster,
        param_grid,
        y,
        X,
        verbose=0,
        n_jobs=1,
    ):
        """
        Iterate over all possible combinations of hyperparameter values,
        evaluate the corresponding forecast models based on the time
        series `y` and returns the exhaustive ranking.

        Parameters
        ----------
        forecaster : BaseForecaster
            Forecaster model to be tuned
        param_grid : dict (key = hyperparameter name, value = array of
            hyperparameter values). Hyperparameter names and their values
            to be explored.
        y : array-like, shape (n_timesteps x n_endo)
            Time series which will be split into training/validation pieces.
        X : array-like or function, shape (n_timesteps x n_exo), optional
            (default=None). Exogeneous time series to adjust to. Can also
            be understood as a control signal or some externally available
            information.

        Returns
        -------
        grid.cv_results_ :
            Sklearn-compatible representation of the grid search results.
            It contains the whole ranking of all the combinations of
            hyperparameter values, corresponding metric values and
            the time consumed during the training and validation stages.
        """
        if X is None:
            # TODO: some sklearn subroutines like gridsearchcv
            # cannot pass X=None so we need to pass array of None
            X = [None for _ in range(len(y))]

        if issubclass(type(forecaster), BaseEstimator):
            param_grid = {
                    f'regressor__custom_estimator__{k}': v for k, v in param_grid.items()
                    }
        scorer = make_scorer(mae, greater_is_better=False)
        grid = GridSearchCV(
                    forecaster,
                    param_grid,
                    scoring=scorer,
                    cv=TimeSeriesSplit(
                        n_splits=self.n_splits,
                        test_size=self.n_test_timesteps
                        ),
                    verbose=verbose,
                    n_jobs=n_jobs,
                    pre_dispatch=n_jobs,
                    )
        grid.fit(X=X, y=y)
        return grid

    def score(self, y_true, y_pred, y_origin=None, log=True):
        score = []
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        if y_origin is None:
            y_origin = y_true
        if len(y_origin.shape) > 1:
            y_origin = y_origin[:, 0]
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, 0]
        y_origin = y_origin.flatten()
        y_pred = y_pred.flatten()
        for m in self.metric:
            try:
                score.append(m(y_pred=y_pred, y_true=y_origin))
            except:
                score.append(None)
        return score
