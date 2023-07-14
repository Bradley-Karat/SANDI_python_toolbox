import numpy as np

def my_murdaycotts(Delta,delta,r,D0,bb):


# [mlnS, mlnSneuman, mlnSnp, bardelta] = murdaycotts(Delta,delta,r,D0,g)

# calculates diffusion attenuation, mlnS = - ln(S/S0), 
# inside a perfectly reflecting sphere of radius r, free diffusion
# coefficient D0, bvalue b (in IS units of s/m), with pulse width delta
# and distance Delta between the fronts of the pulses,
# according to Murday and Cotts, JCP 1968

# Reference value: g = 0.01070 for 40 mT/m

# Here, bardelta = delta/td, parameter of applicability of Neuman's
# approximation:
# for bardelta>>1, Neuman's limit mlnSneuman can be used, independent of Delta.
# for bardelta<<1, narrow pulse limit mlnSnp can be used.

# (c) Dmitry Novikov, June 2021

    Delta = np.divide(Delta,1000) # in s
    if Delta==0:
        Delta = 1e-9

    delta = np.divide(delta,1000) # in s
    if delta==0:
        delta = 1e-9

    r = r # in microns
    D0 = D0 # % in microns^2/ms
    bb = bb

    GAMMA = 2.675987e8

    ginput = np.sqrt( bb*1e9/(Delta - delta/3)) / ((GAMMA*delta))

    g = ginput*GAMMA*1e-9; # in 1/micron*ms

    td = r**2/D0
    bardelta = [delta*1000/td]
    barDelta = [Delta*1000/td]

# precompute beta_{1,k}, zeros of x dJ_{3/2}(x)/dx = 1/2 * J_{3/2}(x)
    N = 20 # max # of terms in the sum
# dJ = @(x) besselj(3/2,x) - x.*(besselj(1/2,x)-besselj(5/2,x));
# beta = @(k) fzero(dJ, [(k-1)*pi+eps, k*pi]); 
# b = zeros(1,N); for k=1:N, b(k) = beta(k); end 
# One can now tabulate b(k) and never calculate them again: 
    b = [2.0816,    5.9404,    9.2058,   12.4044,   15.5792,   18.7426,   21.8997,  25.0528,   28.2034,   31.3521,   34.4995,   37.6460,   40.7917,   43.9368,   47.0814,   50.2257,   53.3696,   56.5133,   59.6567,   62.8000]
    mlnS = 0

    for k in range(N):

        if len(bardelta)==1:
    
            mlnS = mlnS + (2/(b[k]**6*(b[k]**2-2)))*(-2 + 2*b[k]**2*bardelta[0] + 
                2*(np.exp(-b[k]**2*bardelta[0])+np.exp(-b[k]**2*barDelta[0])) - 
                np.exp(-b[k]**2*(bardelta[0]+barDelta[0])) - np.exp(-b[k]**2*(barDelta[0]-bardelta[0]))) 
              
        else:
        
            mlnS = mlnS + (2/(b[k]**6*(b[k]**2-2)))*(-2 + 2*b[k]**2*bardelta[k] + 
                    2*(np.exp(-b[k]**2*bardelta[k])+np.exp(-b[k]**2*barDelta[k])) -
                    np.exp(-b[k]**2*(bardelta[k]+barDelta[k])) - np.exp(-b[k]**2*(barDelta[k]-bardelta[k])))
              

    mlnS = mlnS*D0*g**2*td**3

    mlnSneuman = (16/175)*g**2*delta*r**4/D0

    mlnSnp = (g*delta)**2*r**2/5

    return mlnS # mlnSneuman, mlnSnp, bardelta, b