from .sigmoid import sigmoid
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    OTx = np.dot(theta, X.T)
    g = sigmoid(OTx)
    #Prediction: 1 for g >= 0.5 0 otherwise
    preds = np.where(g >= 0.5, 1, 0)
    
    # E.g.
    #g   : 0.1 0.8 0.3 0.6
    #pred: 0.0 1.0 0.0 1.0
    #y   : 0.0 1.0 0.0 0.0  3/4 Hit ==> accuracy 75%
    
    if (y.any != None):
        m = y.shape[0]
        accuracy = np.sum(np.where(y == preds, 1,0)) / m

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy