import numpy as np
import time
from matplotlib import pyplot as plt

def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [n]

    """

    centers = None
    assign = np.zeros(len(X))

    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the input data X and store the         #
    # resulting cluster-centers as well as cluster assignments.           #
    #                                                                     #
    #######################################################################

    # 1st step: Chose k random rows of X as initial cluster centers
    centers = X[np.random.choice(np.size(X,0),k,replace=False)]
    
    m = np.size(X,0) # number of data
    
    sum = 0 # for convergence check
    for i in range(max_iter):
        prev_assign = assign

        # 2nd step: Update the cluster assignment
        assign = np.array([np.argmin(np.linalg.norm(X[i] - centers,axis=1)) for i in range(m)])

        # 3rd step: Check for convergence
        newSum = np.sum(np.power(np.linalg.norm(X - centers[assign], axis=1), 2))
        if(np.isclose(sum, newSum)):
            break
        sum = newSum
 
        # 4th step: Update the cluster centers based on the new assignment
        centers = np.array([np.sum(X[assign==i], axis=0) / np.sum(assign==i) for i in range(k)])

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(i+1, exec_time))

    return centers, assign
