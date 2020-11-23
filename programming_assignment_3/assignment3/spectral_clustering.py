import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def apply_kernel(D, c=0.01):
    """
    Applies radial-kernel to a given pairwise distancies matrix W 

    Args:
        D: The pairwise distancies matrix of shape [n, n]

    Returns:
        W: Matrix of shape [n, n]
        
    """
    
    W = None
    
    #######################################################################    
    # TODO:                                                               #
    # Apply radial-kernel to D using the formula from the corresponding   #
    # section of the notebook                                             #
    #                                                                     #
    #######################################################################
    
    pass
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return W


def compute_laplacian(X):
    """
    Compute Laplacian of the similarity graph of X

    Args:
        X: The data of shape [n, num_features]

    Returns:
        L: Laplacian of shape [n, n]
        
    """
    
    L = None
    
    #######################################################################
    # TODO:                                                               #
    # 1) Compute NxN matrix D of pairwise distances                       #
    # 2) Apply radial-kernel to D to get egde weights' matrix W           #
    # 3) Compute diagonal matrix G of nodes' degrees                      #
    # 4) Compute L = G - W                                                #
    #                                                                     #
    #######################################################################
    
    pass
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return L


def spectral_transform(X, k):
    """
    Compute matrix Z of k eigenvalues of the Laplacian of the similarity graph of X

    Args:
        X: The data of shape [n, num_features]
        k: The number of eigenvectors with smallest eigenvalues

    Returns:
        Z: A matrix of shape [n, k] with k eigenvectors corresponding to k smallest eigenvalues of L in columns
        
    """
    
    L = compute_laplacian(X)
    
    #######################################################################
    # TODO:                                                               #
    # 1) Compute eigenvalues and eigenvectors of L using np.linalg.eig()  #
    # 2) Choose k eigenvectors corresponding to smallest eigenvalues      #
    #                                                                     #
    #######################################################################
    
    pass
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return Z
    

def spectral_clustering(X, k):
    """
    Perform spectral clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        Z: Eigenvectors matrix of shape [n, k]
        assign: A vector of cluster assignments for each example in X of shape [n] 
        
    """
    
    start = time.time()
    
    Z = spectral_transform(X, k)
    
    km = KMeans(n_clusters=k)
    assign = km.fit_predict(Z)
    
    exec_time = time.time()-start
    print('Execution time: {}s'.format(exec_time))
    
    return Z, assign