import numpy as np
import time

def apply_MLP_python(signal,trainedML,noMLdebias,method,log):
    
    # Apply pretrained Multi Layer Perceptron regressor

    # Author:
    # Dr. Marco Palombo
    # Cardiff University Brain Research Imaging Centre (CUBRIC)
    # Cardiff University, UK
    # 8th december 2021
    # Email: palombom@cardiff.ac.uk

    # Ported into python by Bradley Karat
    # University of Western Ontario
    # 20th June 2023
    # Email: bkarat@uwo.ca

    tic = time.time()

    Mdl = trainedML['Mdl']
    Slope = trainedML['Slope']
    Intercept = trainedML['Intercept']


    if method == 1: #train a single MLP to predict all model parameters

        mpgMean = np.zeros((signal.shape[0],5,len(Mdl)))

        for j in range(len(Mdl)):
            net = Mdl[j]
            mpgMean[:,:,j] = net.predict(signal)
            if not noMLdebias:
                for i in [0,2,4]:
                    mpgMean[:,i,j] = (mpgMean[:,i,j] - Intercept[i,j]/Slope[i,j])

        mpgMean = np.mean(mpgMean,axis=2)
        
    else: #train a single MLP to predict a single model parameter

        nrow = len(Mdl)
        ncol = len(Mdl[0])
        mpgMean = np.zeros((signal.shape[0],nrow,ncol))

        for j in range(ncol):
            for i in range(nrow):
                net = Mdl[i][j]
                mpgMean[:,i,j] = net.predict(signal)
                if not noMLdebias:
                    if i==0 or i==2 or i==4:
                        mpgMean[:,i,j] = (mpgMean[:,i,j] - Intercept[i,j]/Slope[i,j])

        mpgMean = np.mean(mpgMean,axis=2)

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - MLP fitted in {round(tottime)} sec.')
    f.close()

    return mpgMean

