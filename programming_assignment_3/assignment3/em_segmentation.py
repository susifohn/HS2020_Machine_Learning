import numpy as np
from sklearn.mixture import GaussianMixture

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """

    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #
    #      generate the label-image                                       #
    #######################################################################
    
    # 1st: Augment the pixel features with their 2D coordinates to get...
    
    #Get width and height of image
    shape = np.shape(img)
    h = shape[0]
    w = shape[1]

    #Pixels in the image are associated with a point in the square [-1,1] x [-1,1] 
    height = np.linspace(-1,1,h)
    width = np.linspace(-1,1,w)

    #Points of image
    xx, yy = np.meshgrid(width, height)

    #RGBXY (vec)tor, to match the pixel colour in the imgage with the coordinates in [-1,1]x[-1,1]
    img_vec = np.array([np.append(img[i,j], np.array([xx[i,j], yy[i,j]])) for j in range(w) for i in range(h)])

    # min and max color value
    min_rgb = np.min(img_vec[:,0:3])
    max_rgb = np.max(img_vec[:,0:3])
    
    # interpolate lin. each  color vectr to some value in the volume [-1,1]x[-1,1]x[-1,1]
    img_vec[:,0:3] = np.interp(img_vec[:,0:3], (min_rgb, max_rgb), (-1,1))
    
    # 2nd: Fit the MoG to the resulting data using...
    
    gm = GaussianMixture(n_components =k, max_iter = max_iter)
    
    # 3rd: Predict the assignment of the pixels to the gaussian and...
    
    labels = gm.fit_predict(img_vec)
    label_img = np.transpose(np.reshape(labels, (w,h)))   


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
