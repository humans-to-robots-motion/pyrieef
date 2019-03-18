#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                           Jim Mainprice on Sunday May 17 2015

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
# from sklearn.externals.six.moves import xrange
# from sklearn.cluster import KMeans
# from gmr_simple import GMM_GMR


def plot_gmm(mean, covar, plot):
    print "mean : ", mean
    print "covar : \n", covar
    v, w = linalg.eigh(covar)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    coeff = 3  # 2.4477 = 95% ellipse
    ell = mpl.patches.Ellipse(
        mean,
        coeff * np.sqrt(v[0]), coeff * np.sqrt(v[1]),
        180 + angle,
        color=color)
    ell.set_clip_box(plot.bbox)
    ell.set_alpha(0.5)
    plot.add_artist(ell)
    plot.axis([-.15, 0.1, -.06, 0.06])


if __name__ == '__main__':

    X = np.loadtxt(open("data/data2_a.csv", "rb"), delimiter=",")
    X = X.transpose()
    X *= 1.  # Weird does not converge without scaling
    print X.shape
    Y = np.loadtxt(open("data/data2_b.csv", "rb"), delimiter=",", skiprows=1)

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    splot1 = plt.subplot(2, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], .8, color='r')
    plt.scatter(X[:, 2], X[:, 3], .8, color='g')

    clf = mixture.GMM(
        n_components=3,
        min_covar=1e-8,
        covariance_type='full',
        n_iter=100)
    clf.fit(X)

    Y_ = clf.predict(X)

    # Does not work
    # y_pred, y_covar = clf.predict_gmr(X)

    # Plot an ellipse to show the Gaussian component

    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        print "___________________"
        print "plot_gmm : ", i
        print "mean : ", mean
        print "covariance : \n", covar

        print "**********"
        splot2 = plt.subplot(2, 1, 2)

        plot_gmm(mean[0:2], covar[0:2, 0:2], splot2)
        plot_gmm(mean[2:4], covar[2:4, 2:4], splot2)

        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        plt.scatter(X[Y_ == i, 2], X[Y_ == i, 3], .8, color=color)

    plt.scatter(X[Y_ == i, 2], X[Y_ == i, 3], .8, color=color)

    # plt.ylim(-0.6, 0.6)
    # plt.xlim(-1., 1.)

    plt.show()
