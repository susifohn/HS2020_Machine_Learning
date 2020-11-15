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
    """
    #The inverse of the hyper-parameter
    lamb = 1 / float(C);
    
    #the amount of training data
    num_data = np.size(X,0)
    
    #the norm squared of w
    norm = np.dot(w,w)
    
    #sum of weights
    weights = np.dot(X,w)
    weights = weights + b
    weights = np.multiply(np.reshape(weights,np.shape(y)),y)
    weights = 1 - weights
    weights = np.maximum(np.zeros(np.shape(y)),weights)
    weights = np.sum(weights)   
    #l = lamb / 2 * norm + float(weights) / num_data
    
    """
    # *******************************************
    m = np.size(X,0)
    
    f_x = np.dot(X,w) +b
    chi = 1 - y * f_x

    max = chi[chi >= 0] # max(0, chi)
    sum = np.sum(max)
    
    l = 1/(2.0 * C) * np.dot(w,w) + 1/m * sum
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
