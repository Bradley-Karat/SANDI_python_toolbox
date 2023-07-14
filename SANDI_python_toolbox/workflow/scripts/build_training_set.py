import numpy as np
from drchrnd import drchrnd
import time
from RiceMean import RiceMean
from my_murdaycotts import my_murdaycotts

def build_training_set(bvals,bvecs,output,modeldict,log):

# Builds the dataset for supervised training of the machine learning models
# for SANDI fitting.

# Author:
# Dr. Marco Palombo
# Cardiff University Brain Research Imaging Centre (CUBRIC)
# Cardiff University, UK
# 4th August 2022
# Email: palombom@cardiff.ac.uk

# Ported into python by Bradley Karat
# University of Western Ontario
# 20th May 2023
# Email: bkarat@uwo.ca


    tic = time.time()

    ## Build the training set

    bval_filename = bvals
    bvec_filename = bvecs
    output_folder = output

    modeldict = modeldict

    Nparams = modeldict['Nparams']
    sigma_mppca = modeldict['sigma_mppca']
    sigma_SHresiduals = modeldict['sigma_SHresiduals']
    Nset = modeldict['Nset']
    delta = modeldict['delta']
    smalldel = modeldict['smalldel']
    paramsrange = modeldict['paramsrange']
    log = log

    params = np.zeros((Nset,Nparams))

    T = drchrnd([1,1,1],np.round(Nset))
    params[:,0:2] = T[:,0:2]


# Sample the other model parameters from a uniform distribution

    for i in range(2,Nparams):
        params[:,i] = (paramsrange[i,1]-paramsrange[i,0])*np.random.uniform(size=(Nset)) + paramsrange[i,0]

    Dis = modeldict['Dsoma']

    fstick = lambda p,x,costheta: np.exp(-x*p*costheta**2) #np.exp(-x*p[0]*costheta**2)
    fsphere = lambda p,x: np.exp(-(my_murdaycotts(delta,smalldel,p,Dis,x)))
    fball = lambda p,x: np.exp(-p*x)

    fcomb = lambda p,x,costheta: p[0]*fstick(p[2],x,costheta) + p[1]*fsphere(p[3],x) + (1-p[0]-p[1])*fball(p[4],x)

    # Load bvals and bvecs
    bvals = np.loadtxt(bval_filename)
    bvals = np.round(bvals/100)*100
    bvals[bvals==0] = 1e-6
    bunique = np.unique(bvals)

    Ndirs_per_shell = np.zeros((len(bunique),1))
    for i in range(len(bunique)):
        Ndirs_per_shell[i] = np.sum(bvals == bunique[i])

    bvecs = np.loadtxt(bvec_filename)
    bvecs = bvecs/np.linalg.norm(bvecs) #normalize bvecs
    bvecs[np.isnan(bvecs)] = 0

    f = open(log, "a")
    f.write(f"\nGenerating {Nset} random fibre directions.")
    f.close()

    phi = np.random.rand(Nset,1)*2*np.pi # *2pi to get directions across whole sphere
    u = 2*np.random.rand(Nset,1)-1 # cos(theta): ranging from -1 and +1
    term1 = np.sqrt(1-u**2)*np.cos(phi)
    term2 = np.sqrt(1-u**2)*np.sin(phi)

    fibre_orientation = np.transpose(np.concatenate((term1,term2,u),axis=1))

    costheta = np.zeros((Nset,len(bvals)))

    f = open(log, "a")
    f.write(f"\nCalculating angles between fibres and gradients.")
    f.close()

    for i in range(Nset):
        simfibre = np.tile(fibre_orientation[:,i],[len(bvals),1])
        costheta[i,:] = (simfibre*np.transpose(bvecs)).sum(1) #np.dot(simfibre,bvecs) - element-wise multiplication then sum across columnn

    database_dir = np.zeros((Nset, len(bvals)))
    database_dir_with_rician_bias = np.zeros((Nset, len(bvals)))
    database_dir_with_rician_bias_noisy = np.zeros((Nset, len(bvals)))

    f = open(log, "a")
    f.write(f"\nCalculating signals per diffusion gradient direction and add Rician bias following sigma distribution from MPPCA, with median SNR = {np.nanmedian(1/sigma_mppca)} to the signal for each direction.")
    f.close()

    f = open(log, "a")
    f.write(f"\nAdding Gaussian noise following the distribution from SH residuals, with median SNR = {np.nanmedian(1/sigma_SHresiduals)} to the signal for each direction.")
    f.close()
    
    for i in range(Nset):
        database_dir[i,:] = fcomb(params[i,:], bvals/1000, costheta[i,:])
        database_dir_with_rician_bias[i,:] = RiceMean(database_dir[i,:], sigma_mppca[i])
        database_dir_with_rician_bias_noisy[i,:] =  database_dir_with_rician_bias[i,:] + sigma_SHresiduals[i]*np.random.normal(0,1,size=len(database_dir_with_rician_bias[i,:]))

    database_train_noisy = np.zeros((params.shape[0], np.size(bunique)))
    database_train = np.zeros((params.shape[0], np.size(bunique)))

    # Identify b-shells and direction-average per shell
    f = open(log, "a")
    f.write(f"\nDirection-averaging the signals.")
    f.close()

    for i in range(np.size(bunique)):
        database_train_noisy[:,i] = np.nanmean(database_dir_with_rician_bias_noisy[:,bvals==bunique[i]],1)
        database_train[:,i] = np.nanmean(database_dir[:,bvals==bunique[i]],1)

    params_train = params

    for i in range(database_train_noisy.shape[1]):
        database_train_noisy[:,i] = database_train_noisy[:,i]/database_train_noisy[:,1]  # Normalize by the b=0

    bunique = bunique/1000

    databasedict = {"database_train":database_train,"params_train":params_train}
    np.save(output_folder,databasedict)

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - Set built in {round(tottime)} sec.')
    f.close()

    return database_train, database_train_noisy, params_train, sigma_mppca, bunique, Ndirs_per_shell
