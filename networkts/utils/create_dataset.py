import numpy as np


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-2*look_back+1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i+look_back:(i+2*look_back), 0])
    return np.array(dataX), np.array(dataY)