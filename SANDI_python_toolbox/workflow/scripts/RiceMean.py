import scipy.special
import numpy as np

def RiceMean(nu,sig):
    # This code is from Jelle Veraart - NYU

    x = -(nu**2) / (2*sig**2)
    #scale = np.exp(-np.abs(x/2)) #in matlab implementation the bessel function is scaled and de-scaled,
    #besseli(0, x/2, 1).*exp(abs(real(x/2))) where the third input into besseli (1) does the following:
    #besseli(0, x/2, 1) / exp(abs(real(x/2))).*exp(abs(real(x/2))) = besseli(0, x/2, 1)

    I0 = scipy.special.iv(0,x/2) #python does not scale by default
    I1 = scipy.special.iv(1, x/2)

    K = np.exp(x/2)*(x*I1  + (1-x)*I0)

    mu = 1.2533 * sig * K

    nanlocs = np.nonzero(np.isnan(mu) | ~np.isfinite(mu))[0]
    mu[nanlocs] = nu[nanlocs]
    mu = mu/mu[0] # Provide signal normalized by the b=0 signal

    return mu