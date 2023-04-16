from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def create_lissagy_sin_cos(A, a, d, B, b, t_series, sublot):
    x = A * np.sin(a * t_series + d)
    y = B * np.cos(b * t_series)
    plt.subplot(sublot)
    plt.plot(x, y)


def create_lissagy_sin_sin(A, a, d, B, b, t_series, subplot):
    x = A * np.sin(a * t_series + d)
    y = B * np.cos(b * t_series)
    plt.subplot(subplot)
    plt.plot(x, y)


pi_series = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
t_series = np.arange(1, 100, 0.001)

plt.grid(True)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xticks(np.linspace(-1, 1, 11))
plt.yticks(np.linspace(-1, 1, 11))

create_lissagy_sin_sin(1, 1, 0, 1, 1, t_series, 241)
create_lissagy_sin_sin(1, 1, 10, 1, 1, t_series, 242)
create_lissagy_sin_sin(1, 1, np.pi / 2, 1, 1, t_series, 245)
create_lissagy_sin_sin(1, 1, -np.pi / 2, 1, 1, t_series, 246)

create_lissagy_sin_sin(1, 1, 0, 1, 1, t_series, 122)
create_lissagy_sin_sin(1, 1, 10, 1, 1, t_series, 122)
create_lissagy_sin_sin(1, 1, np.pi / 2, 1, 1, t_series, 122)
create_lissagy_sin_sin(1, 1, -np.pi / 2, 1, 1, t_series, 122)

plt.show()
