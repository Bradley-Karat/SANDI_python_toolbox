import numpy as np
import math
import scipy

def llsFitSH(dirs,y,order): 
#Author: Jelle Veraart  and Santiago Coelho (jelle.veraart@nyulangone.org) 
#copyright NYU School of Medicine, 2022

    X = getSH(order, dirs, 0)
    coef = np.linalg.lstsq(X,y,rcond=None)
    
    return(coef,X)

def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def legendre(n,X) :
    res = []
    n=int(n)
    for m in range(n+1):
        res.append(scipy.special.lpmv(m,n,X))
    return res


def getSH(Lmax, dirs, CS_phase=0):

    # Ylm_n = get_even_SH(dirs,Lmax,CS_phase)
    
    # if CS_phase=1, then the definition uses the Condon-Shortley phase factor
    # of (-1)^m. Default is CS_phase=0 (so this factor is ommited)
    #
    # By: Santiago Coelho (https://github.com/NYU-DiffusionMRI/SMI/blob/master/SMI.m)
            
    if dirs.shape[1]!=3:
        dirs=np.transpose(dirs)

    Nmeas=dirs.shape[0]
    [PHI,THETA,r]=cart2sph(dirs[:,0],dirs[:,1],dirs[:,2]) 
    THETA=np.pi/2-THETA
    l=np.arange(0,Lmax+2,2)

    l_all = np.empty((0))
    m_all = np.empty((0))
    
    for ii in range(len(l)):
        hold_l = l[ii]*np.ones((1,2*l[ii]+1))
        l_all = np.concatenate((l_all,hold_l[0]))
        hold_m = np.arange(-l[ii],l[ii]+1,1)
        m_all = np.concatenate((m_all,hold_m))

    term1 = np.divide((2*l_all+1),(4*np.pi))
    hold_term2 = l_all-np.abs(m_all)

    for jj in range(len(hold_term2)):
        hold_term2[jj] = math.factorial(int(hold_term2[jj]))

    term2 = np.multiply(term1,hold_term2)
    K_lm_hold = l_all+np.abs(m_all)

    for jj in range(len(K_lm_hold)):
        K_lm_hold[jj] = math.factorial(int(K_lm_hold[jj]))
    
    K_lm = np.sqrt(np.divide(term2,K_lm_hold))

    if CS_phase==0:
        extra_factor=np.ones((K_lm.shape))
        extra_factor[m_all!=0]=np.sqrt(2)
    else:
        extra_factor=np.ones((K_lm.shape))
        extra_factor[m_all!=0]=np.sqrt(2)
        extra_factor=np.power(np.multiply(extra_factor,(-1)),m_all)

    P_l_in_cos_theta=np.zeros((np.max(l_all.shape),Nmeas))
    phi_term=np.zeros((np.max(l_all.shape),Nmeas))
    id_which_pl=np.zeros((1,np.max(l_all.shape)))

    for ii in range(len(l_all)):
        all_Pls=legendre(l_all[ii],np.cos(THETA))
        all_Pls = np.asarray(all_Pls)
        P_l_in_cos_theta[ii,:]=all_Pls[int(np.abs(m_all[ii])),:]
        id_which_pl[:,ii]=np.abs(m_all[ii])
        if m_all[ii]>0:
            phi_term[ii,:]=np.cos(m_all[ii]*PHI)
        elif m_all[ii]==0:
            phi_term[ii,:]=1
        elif m_all[ii]<0:
            phi_term[ii,:]=np.sin(-m_all[ii]*PHI)
    
    term1 = np.transpose(np.tile(extra_factor,(Nmeas,1)))
    hold_term2 = np.transpose(np.tile(K_lm,(Nmeas,1)))
    term2 = np.multiply(term1,hold_term2)
    term3 = np.multiply(term2,phi_term)
    Y_lm = np.multiply(term3,P_l_in_cos_theta)
    Y=np.transpose(Y_lm)
    return Y
