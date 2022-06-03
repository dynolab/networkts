import tensorflow as tf
import numpy as np


def make_ds(
        df: np.array or list,
        size: int,
        batch_size: int = 64,
        shift: int = 1,
        drop: int = 0,
        skip: int = 0
        ):
    if drop != 0:
        data = df[:-drop]
    else:
        data = df
    ds = tf.data.Dataset.from_tensor_slices(data[skip:])
    ds = ds.window(size, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(size))
    return ds.batch(batch_size, drop_remainder=True)
