import numpy as np
from sttf.utils.create_dataset import create_dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from sttf.base import BaseForecaster
from sttf.utils.create_dataset import create_dataset


class NtsLstm(BaseForecaster):
    def __init__(
                self,
                lstm_units=64,
                drop=0.1,
                look_back=1,
                epochs=100,
                batch_size=32
                ):
        self.lstm_units = lstm_units
        self.drop = drop
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size

        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super().__init__()

    def _fit(self, y, X):
        # y needs be more than test_size (n_timesteps) in 2 times

        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(1, self.look_back)))
        model.add(Dense(self.look_back))
        model.compile(loss='mse', optimizer='adam', metrics=['mape', 'mae'])

        '''
        model = Sequential()
        model.add(LSTM(
                    units=self.lstm_units,
                    return_sequences=True,
                    input_shape=(1, self.look_back)
                    ))
        model.add(Dropout(rate=self.drop))
        model.add(LSTM(units=self.lstm_units, return_sequences=True))
        model.add(Dropout(rate=self.drop))
        model.add(LSTM(units=self.lstm_units, return_sequences=True))
        model.add(Dropout(rate=self.drop))
        model.add(LSTM(units=self.lstm_units, return_sequences=False))
        model.add(Dropout(rate=self.drop))
        model.add(Dense(self.look_back))
        model.compile(loss='mse', optimizer='adam', metrics=['mape', 'mae'])
        '''

        self._y = y
        train_x, train_y = create_dataset(
                                dataset=y.reshape(-1, 1),
                                look_back=self.look_back
                                )
        train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

        model.fit(
                train_x,
                train_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=False
                )
        self.model = model
        self._is_fitted = True
        return self

    def _predict(self, n_timesteps, X=None):
        y = np.array(self._y)
        y = y[-n_timesteps:]
        y = y.reshape(1, 1, n_timesteps)
        y_pred = self.model.predict(y).reshape(-1)
        return y_pred

    def _update(self, y, X):
        self.model = self.fit(self, y=y, X=X)
        return self
