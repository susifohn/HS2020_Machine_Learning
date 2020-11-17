import numpy as np


def svm_loss(w, b, X, y, C):
    """
    Computes the loss of a linear SVM w.r.t. the given data and parameters

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]
        C: SVM hyper-parameter

    Returns:
        l: The value of the objective for a linear SVM

    """

    l = 0
    #######################################################################
    # TODO:                                                               #
    # Compute and return the value of the unconstrained SVM objective     #
    #                                                                     #
    #######################################################################
    
    m = np.size(X,0)
    
    f = np.dot(X,w) +b
    xsi = 1 - y * f

    max = xsi[xsi >= 0] # max(0, chi)
    
    l = 1/(2*C) * np.dot(w,w) + 1/m * np.sum(max)
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
