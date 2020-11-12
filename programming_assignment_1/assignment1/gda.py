from .cost_function import cost_function
import numpy as np
import time


def gda(X, y):
    """
    Perform Gaussian Discriminant Analysis.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]

    Returns:
        theta: The value of the parameters after logistic regression

    """

    theta = None
    phi = None
    mu_0 = None
    mu_1 = None
    sigma = None

    X = X[:, 1:]    # Note: We remove the bias term!
    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Perform GDA:                                                        #
    #   - Compute the values for phi, mu_0, mu_1 and sigma                #
    #                                                                     #
    #######################################################################

    #see where y = 0, then 1, 0 otherwise
    y_0 = np.where(y == 0, 1, 0)
    m = y.shape[0]
    x_num = X.shape[0] # num training sets
    #count ones in y
    y_ones = np.sum(y)
    y_zeros = np.sum(y_0)
    
    phi = y_ones / m #probability y=1
    mu_0 = np.dot(y_0, X) / y_zeros
    mu_1 = np.dot(y, X) / y_ones
    
    mu_y = np.outer(-y,mu_0) + np.outer(y, mu_1)
   
    sigma = np.dot(( X-mu_y ).T, X-mu_y ) / x_num

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    # Compute theta from the results of GDA
    sigma_inv = np.linalg.inv(sigma)
    quad_form = lambda A, x: np.dot(x.T, np.dot(A, x))
    b = 0.5*quad_form(sigma_inv, mu_0) - 0.5*quad_form(sigma_inv, mu_1) + np.log(phi/(1-phi))
    w = np.dot((mu_1-mu_0), sigma_inv)
    theta = np.concatenate([[b], w])
    exec_time = time.time() - start

    # Add the bias to X and compute the cost
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    loss = cost_function(theta, X, y)

    print('Iter 1/1: cost = {}  ({}s)'.format(loss, exec_time))

    return theta, None
