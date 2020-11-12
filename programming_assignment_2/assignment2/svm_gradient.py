import numpy as np


def svm_gradient(w, b, x, y, C):
    """
    Compute gradient for SVM w.r.t. to the parameters w and b on a mini-batch (x, y)

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        x: A mini-batch of training example [k, num_features]
        y: Labels corresponding to x of size [k]

    Returns:
        grad_w: The gradient of the SVM objective w.r.t. w of shape [k, num_features]
        grad_v: The gradient of the SVM objective w.r.t. b of shape [k, 1]
        Corr ref mail from 9.11. TOTO add to Git...
        grad_w: The gradient of the SVM objective w.r.t. w of shape [num_features]
        grad_b: The gradient of the SVM objective w.r.t. b of shape [1]

    """

    grad_w = np.zeros(x.shape[1])
    grad_b = 0


    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of w and b.            #
    # Compute the partial derivatives and set grad_w and grad_b to the    #
    # partial derivatives of the cost w.r.t. both parameters              #
    #                                                                     #
    #######################################################################
    
    #compute the lambda value
    lamb = 1 / float(C)
    
    #scale w by lambda
    scalar_w = lamb * w
    
    #get the amount of data in the training batch
    num_data = np.size(x, 0)
    
    #reshape y so it can be used in the next calculation
    y = np.reshape(y,(num_data,1))
  
    #calculate the sum of the data scaled by its corresponding label
    weights = np.ravel(np.dot(x.T,y))
    
    #compute the gradient with respect to w
    grad_w = scalar_w + weights / num_data
    
    #compute the gradient with respect to b
    grad_b = np.sum(y) / num_data

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad_w, grad_b
