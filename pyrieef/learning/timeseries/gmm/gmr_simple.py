#!/usr/bin/env python
"""
Copyright (c) June, 2014, Eric Wilkinson
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from sklearn import mixture
import numpy as np


class GMM_GMR(mixture.GMM):
    """
    Class that extends from the sklearn GMM to include Gaussian Model Regression

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.
    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance
    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.
    thresh : float, optional
        Convergence threshold.
    n_iter : int, optional
        Number of EM iterations to perform.
    n_init : int, optional
        Number of initializations to perform. the best results is kept
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.
    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.
    Attributes
    ----------
    `weights_` : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.
    `means_` : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.
    `covars_` : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::
            (n_components, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.
    See Also
    --------
    GMM : Finite Gaussian mixture model fit with EM
    """

    def __init__(self, n_components=1, covariance_type='full',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc'):
        super(GMM_GMR, self).__init__(n_components, covariance_type,
                                      random_state=random_state,
                                      tol=thresh, min_covar=min_covar,
                                      n_iter=n_iter, params=params,
                                      init_params=init_params)

    def predict_gmr(self, X):
        """
        Use gaussian model regression to return predicted values.
        Currently only works for 1-dimentional prediction.

        Parameters
        ----------
        X : array-lke, Input indepedent variables.


        Returns
        ----------
        Y_pred, Y_covars : array-like, Prediction and covariance for each input variable

        """

        # Algorithm implemented by Sylvain Calinon
        # %% Slow one-by-one computation (better suited to understand
        # the algorithm)
        #
        #  1) Compute the influence of each GMM component, given input x
        #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #  for i=1:nbStates
        #    Pxi(:,i) = gaussPDF(x, Mu(in,i), Sigma(in,in,i));
        #  end
        #  beta = (Pxi./repmat(sum(Pxi,2)+realmin,1,nbStates))';
        #  2) Compute expected output distribution, given input x
        #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #  y = zeros(length(out), nbData);
        #  Sigma_y = zeros(length(out), length(out), nbData);
        #  for i=1:nbData
        #    % 3) Compute expected means y, given input x
        #    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #    for j=1:nbStates
        #      yj_tmp = Mu(out,j) + Sigma(out,in,j) * inv(Sigma(in,in,j)) *
        #               (x(:,i)-Mu(in,j));
        #      y(:,i) = y(:,i) + beta(j,i).*yj_tmp;
        #    end
        #    % 4) Compute expected covariance matrices Sigma_y, given input x
        #    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #    for j=1:nbStates
        #      Sigmaj_y_tmp = Sigma(out,out,j) -
        #             (Sigma(out,in,j) * inv(Sigma(in,in,j)) * Sigma(in,out,j));
        #      Sigma_y(:,:,i) = Sigma_y(:,:,i) + beta(j,i)^2. * Sigmaj_y_tmp;
        #    end
        #  end

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Expand the input variables so that each can be computed
        # independently for every mixture component
        X_expanded = X[:, None].repeat(self.n_components, axis=1)
        diff_mean = (X_expanded - self.means_[:, 0])

        # Conditional Expectation for each individual component
        # (i.e. what each gaussian believes is the correct prediction).
        # This will be weighted and summed later.
        conditional_expectation = (
            self.means_[:, 1] +
            self.covars_[:, 1, 0] / self.covars_[:, 0, 0] * diff_mean)

        # This will only work for a 1 dimensional prediction Conditional
        # Covariance for each individual component. This will be weighted and
        # summed later.
        conditional_cov = (
            g.covars_[:, 1, 1] -
            self.covars_[:, 1, 0] /
            self.covars_[:, 0, 0] * self.covars_[:, 0, 1])

        priors = np.exp(-0.5 * (diff_mean ** 2) / self.covars_[:, 0, 0])

        # Compute the weight for each mixture guess based on how probable
        # the mixture is at being the correct representative of the input data
        Y_pred = np.empty(X.shape[0])
        Y_covar = Y_pred.copy()

        for i in range(X.shape[0]):
            beta_k = priors[i, :] / np.sum(priors[i, :])
            Y_pred[i] = beta_k.dot(conditional_expectation[i, :])
            Y_covar[i] = (beta_k ** 2).dot(conditional_cov)

        return Y_pred, Y_covar


if __name__ == '__main__':
    num_mixtures = 50

    t = np.arange(0, 10, 0.01)
    noise_y = np.random.random(len(t)) * 0.5
    noise_t = np.random.random(len(t)) * 0.5
    t = t + noise_t
    y = np.cos(t) + noise_y

    Y = np.column_stack((t, y))

    # Number of samples per component
    n_samples = 100

    g = GMM_GMR(num_mixtures, covariance_type='full')
    g.fit(Y)
    y_pred, y_covar = g.predict_gmr(t)

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(t, y)
    # plt.plot(t, y_pred, lw='4')
    plt.plot(t, y_pred)
    plt.show()