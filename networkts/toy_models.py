import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def var3_generator(
    series_size: int = 5000
):
    white_noise = np.random.normal(0, 1, size=(series_size, 4))
    m = np.zeros(series_size)
    m[0] = white_noise[0, 0]
    for j in range(1, series_size):
        m[j] = 0.7*m[j-1] + white_noise[j-1, 0]

    y = np.zeros(series_size)
    y[0] = white_noise[0, 1]
    for j in range(1, series_size):
        y[j] = 0.8*y[j-1] + 0.8*m[j-1] + white_noise[j-1, 1]
    
    x = np.zeros(series_size)
    x[0] = white_noise[0, 2]
    for j in range(1, series_size):
        x[j] = 0.7*x[j-1] - 0.8*y[j-1] + white_noise[j-1, 2]
    
    z = np.zeros(series_size)
    z[:3] = white_noise[:3, 3]
    for j in range(3, series_size):
        z[j] = 0.5*z[j-1] + 0.5*y[j-2] + 0.6*m[j-3] + white_noise[j-1, 3]
    
    return x, y, z, m

def var3_generator_with_season(
    series_size: int = 5000,
    season: int = 10,
):
    white_noise = np.random.normal(0, 1, size=(series_size, 4))
    m = np.zeros(series_size)
    for j in range(1, series_size):
        m[j] = 0.7*m[j-1] + white_noise[j-1, 0]
        if j%season == 0:
            m[j] = abs(m[j]) + 10

    y = np.zeros(series_size)
    for j in range(1, series_size):
        y[j] = 0.8*y[j-1] + 0.8*m[j-1] + white_noise[j-1, 1]
    
    x = np.zeros(series_size)
    for j in range(1, series_size):
        x[j] = 0.7*x[j-1] - 0.8*y[j-1] + white_noise[j-1, 2]
    
    z = np.zeros(series_size)
    for j in range(1, series_size):
        z[j] = 0.5*z[j-1] + 0.5*y[j-2] + 0.6*m[j-3] + white_noise[j-1, 3]
    
    return x, y, z, m

if __name__ == '__main__':
    series_size = 6000
    x, y, z, m = var3_generator_with_season(series_size, 288)
    data = pd.DataFrame(np.array([x, y, z, m]).T, columns=['X', 'Y', 'Z', 'M'], index=range(series_size))

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(18, 10))
    fig.suptitle("Toy VAR(3) process")
    ax[-1].set_xlabel('Time point')
    for j in range(data.shape[1]):
        ax[j].plot(data.iloc[:, j])
        ax[j].set_ylabel(data.columns.values[j], rotation=0)
    plt.savefig('tests/figs/toy_example.png', dpi=200)
    plt.show()

