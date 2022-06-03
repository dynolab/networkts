import numpy as np


def read_score(file_name: str):
    f = open(file_name, "r")
    mape = []
    mae = []
    for row in f.readlines()[:-3]:
        score = row[:-1].split(' ')
        if len(score) == 2:
            mape.append(float(score[0]))
            mae.append(float(score[1]))
    f.close()
    return np.array(mape), np.array(mae)
