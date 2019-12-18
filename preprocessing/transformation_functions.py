"""File Contains Various image Preprocessing Functions"""
import numpy as np
from skimage import transform 


def min_max_scaling(x):
    """
    Standard normalization
    
        argument
            - x: input array
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def rotate_random(array):
    """
       argument 
          - array:  a numpy image array
       return
           -transformed array at random amount 
    
    """
    return transform.rotate(array, np.random.uniform(-30, 30))

def flip_horizontal(array):
    """
        argument
         - array: a numpy img array
         return 
         - horizontally flipped numpy array"""
    return array[:, ::-1]
