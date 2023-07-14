import numpy as np
import time

def apply_RF_python(signal,trainedML,noMLdebias,log):
    
    # Apply pretrained Random Forest regressor

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

    mpgMean = np.zeros((signal.shape[0],len(Mdl)))

    for i in range(len(Mdl)):
        mpgMean[:,i] = Mdl[i].predict(signal)
    if not noMLdebias:
        if i==0 or i==2 or i==4:
            mpgMean[:,i] = (mpgMean[:,i] - Intercept[i]/Slope[i])

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - RF fitted in {round(tottime)} sec.')
    f.close()

    return mpgMean