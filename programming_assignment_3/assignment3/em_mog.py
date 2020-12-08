import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import time


def em_mog(X, k, max_iter=20):
    """
    Learn a Mixture of Gaussians model using the EM-algorithm.

    Args:
        X: The data used for training [n, num_features]
        k: The number of gaussians to be used

    Returns:
        phi: A vector of probabilities for the latent vars z of shape [k]
        mu: A marix of mean vectors of shape [k, num_features]
        sigma: A list of length k of covariance matrices each of shape [num_features, num_features]
        w: A vector of weights for the k gaussians per example of shape [n, k] (result of the E-step)

    """

    # Initialize variables
    mu = None
    sigma = [np.eye(X.shape[1]) for i in range(k)]
    phi = np.ones([k,])/k
    ll_prev = float('inf')
    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Initialize the means of the gaussians. You can use K-means!         #
    #######################################################################

    m = np.size(X,0)
    
    km = KMeans(n_clusters = k).fit(X)
    labels = km.labels_
    
    phi = np.array([np.sum(labels==i) for i in range(k)]) / m
    mu = np.array([np.sum(X[labels==i],axis=0) / np.sum(labels == i) for i in range(k)])
    sigma = np.array([np.dot(np.transpose(X[labels==i] - mu[i]),X[labels==i] - mu[i]) / np.sum(labels==i) for i in range(k)])
    

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    for l in range(max_iter):
        # E-Step: compute the probabilities p(z==j|x; mu, sigma, phi)
        w = e_step(X, mu, sigma, phi)

        # M-step: Update the parameters mu, sigma and phi
        phi, mu, sigma = m_step(w, X, mu, sigma, phi, k)

        # Check convergence
        ll = log_likelihood(X, mu, sigma, phi)
        print('Iter: {}/{}, LL: {}'.format(l+1, max_iter, ll))
        if ll/ll_prev > 0.999:
            print('EM has converged...')
            break
        ll_prev = ll

    # Get stats
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(l+1, exec_time))

    # Compute final assignment
    w = e_step(X, mu, sigma, phi)

    return phi, mu, sigma, w



def log_likelihood(X, mu, sigma, phi):
    """
    Returns the log-likelihood of the data under the current parameters of the MoG model.

    """
    ll = None

    #######################################################################
    # TODO:                                                               #
    # Compute the log-likelihood of the data under the current model.     #
    # This is used to check for convergnence of the algorithm.            #
    #######################################################################
    k = np.size(phi)
    ll = np.sum([np.log(np.sum([multivariate_normal.pdf(x, mean=mu[i] ,cov=sigma[i]) * phi[i] for i in range(k)])) for x in X])

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return ll


def e_step(X, mu, sigma, phi):
    """
    Computes the E-step of the EM algorithm.

    Returns:
        w:  A vector of probabilities p(z==j|x; mu, sigma, phi) for the k
            gaussians per example of shape [n, k]
    """
    w = None

    #######################################################################
    # TODO:                                                               #
    # Perform the E-step of the EM algorithm.                             #
    # Use scipy.stats.multivariate_normal.pdf(...) to compute the pdf of  #
    # of a gaussian with the current parameters.                          #
    #######################################################################

    k = np.size(phi)
    
    w = np.array([ \
                  [multivariate_normal.pdf(x, mean=mu[i] ,cov=sigma[i]) * phi[i] for i in range(k)] \
                  / np.sum([multivariate_normal.pdf(x, mean=mu[i] ,cov=sigma[i]) * phi[i] for i in range(k)]) \
                  for x in X])

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return w


def m_step(w, X, mu, sigma, phi, k):
    """
    Computes the M-step of the EM algorithm.

    """
    #######################################################################
    # TODO:                                                               #
    # Update all the model parameters as per the M-step of the EM         #
    # algorithm.
    #######################################################################

    num_data = np.size(X,0)
    
    dw = np.sum(w,axis=0).astype(float)
    
    phi = np.sum(w,axis=0) / float(num_data)
    mu = np.transpose(np.dot(np.transpose(X),w) / dw)
    sigma = np.array([np.dot(np.transpose(X-mu[i]), (X-mu[i]) * w[:,i][:,np.newaxis]) for i in range(k)]) / dw[:, np.newaxis, np.newaxis]
    

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return phi, mu, sigma
