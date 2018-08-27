import sys
import os
import numpy as np
import matplotlib.pyplot as plt

driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory + os.sep + "..")


def plot_line(line, color, width=2.):
    X = np.array(line)[:, 0]
    Y = np.array(line)[:, 1]
    plt.plot(X, Y, color, linewidth=1.)
    plt.plot(X, Y, color + 'o', linewidth=1.)
    # plt.plot( x_init[0], x_init[1], color + 'o' )
