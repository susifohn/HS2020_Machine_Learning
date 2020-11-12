from .sigmoid import sigmoid
import numpy as np


def gradient_function(theta, X, y):
    """
    Compute gradient for logistic regression w.r.t. to the parameters theta.

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        grad: The gradient of the log-likelihood w.r.t. theta

    """

    grad = None
    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of theta.              #
    # Compute the partial derivatives and set grad to the partial         #
    # derivatives of the cost w.r.t. each parameter in theta              #
    #                                                                     #
    #######################################################################
    
    #Note:OTx means dot(theta.transpose, x)
    #check x is 1d vector, then y, OTx and g must be scalar.
    if (np.ndim(X) == 1):
        OTx = np.dot(theta,X)
        g = sigmoid(OTx)
        grad = X * (g-y) 
    elif (np.ndim(X) == 2): # x is 2d-matrix
        OTx = np.dot(X, theta)
        g = sigmoid(OTx)
        grad = np.sum(np.multiply(X.T,g-y).T, axis=0) #that was somehow tricky

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad