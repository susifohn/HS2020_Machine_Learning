from .sigmoid import sigmoid
import numpy as np


def cost_function(theta, X, y):
    """
    Computes the cost of using theta as the parameter for logistic regression

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        l: The cost for logistic regression

    """

    l = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the log-likelihood l of a particular choice of   #
    # theta.                                                              #
    #                                                                     #
    #######################################################################

    OTx = np.dot(theta,X.T) #Theta times X
   
    l = np.sum(y * np.log(sigmoid(OTx)) + (1-y) * np.log(1-sigmoid(OTx)))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
