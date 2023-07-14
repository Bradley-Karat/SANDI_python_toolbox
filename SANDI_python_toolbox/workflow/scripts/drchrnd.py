import numpy as np

def drchrnd(a,n):

# Draw samples from Dirichlet distribution for volume fractions
    n = int(n)
    p = len(a)
    r = np.random.gamma(np.tile(a,(n,1)),1,size=(n,p))
    r = r/np.transpose(np.tile(np.sum(r,axis=1),(p,1)))
    
    return r

