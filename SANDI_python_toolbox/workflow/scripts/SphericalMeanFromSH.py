import numpy as np
import scipy
from llsFitSH import llsFitSH

def SphericalMeanFromSH(dirs, y, order):
    
    eps = np.finfo(float).eps
    y[y<eps] = eps
    [coef, X] = llsFitSH(dirs,y,order)

    sphericalMean = np.divide(coef[0][0, :],np.sqrt(np.pi*4))
    residuals = X@coef[0] - y
    w = np.ones((residuals.shape))

    n= y.shape[0]
    m = coef[0].shape[0]

    term1 = np.sqrt(np.divide(n,(n-m)))
    term2 = np.multiply(term1,1.4826)
    x = np.multiply(residuals,w)
    term3 = scipy.stats.median_abs_deviation(x)
    sigma = np.multiply(term2,term3)

    return(sphericalMean,sigma)
