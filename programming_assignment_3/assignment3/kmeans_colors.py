from sklearn.cluster import KMeans
import numpy as np
from time import time
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin

def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clusering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the pixel values of the image img.     #
    #######################################################################
    
    h = np.array(img).shape[0] #height
    w = np.array(img).shape[1] #width
    
    img_mx = np.reshape(img, (h*w, 3)).astype(float) #need 2x2 matrix for the img and one dimension for color(rgb)
    
    color_min = np.min(img_mx[:,0:3]) #min color value
    
    color_max = np.max(img_mx[:,0:3]) #max color value
    
    img_mx[:,0:3] = np.interp(img_mx[:,0:3], (color_min, color_max), (-1,1)) #need to normalize color values [-1,1] in 3-dim
    
    km = KMeans(n_clusters = k).fit(img_mx) #use sklearn
    
    labels = km.labels_
    
    labels = np.reshape(labels, (h,w))
    
    centers = km.cluster_centers_
    
    img_cl = np.zeros(np.shape(img))
    for i in range(k):
        img_cl[labels == i] = centers[i] 

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl
