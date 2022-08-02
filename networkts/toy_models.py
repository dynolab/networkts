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

if __name__ == '__main__':
    series_size = 5000
    x, y, z, m = var3_generator(series_size)
    data = pd.DataFrame(np.array([x, y, z, m]).T, columns=['x', 'y', 'z', 'm'], index=range(series_size))

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(18, 10))
    ax[0].plot(x)
    ax[0].set_title('X')
    ax[1].plot(y)
    ax[1].set_title('Y')
    ax[2].plot(z)
    ax[2].set_title('Z')
    ax[3].plot(m)
    ax[3].set_title('M')
    plt.show()

