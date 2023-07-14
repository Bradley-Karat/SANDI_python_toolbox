import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import time

def train_RF_python(database_train, params_train, n_trees, log):

    # Train a Random Forest regressor for SANDI fitting
    
    # Author:
    # Dr. Marco Palombo
    # Cardiff University Brain Research Imaging Centre (CUBRIC)
    # Cardiff University, UK
    # 8th December 2021
    # Email: palombom@cardiff.ac.uk

    # Ported into python by Bradley Karat
    # University of Western Ontario
    # 20th May 2023
    # Email: bkarat@uwo.ca


    tic = time.time()

    #Mdl = np.array((params_train.shape[1],1))
    Mdl = []
    MLprediction = np.zeros((params_train.shape))
    Rsq = np.zeros((params_train.shape[1],1))
    Slope = np.zeros((params_train.shape[1],1))
    Intercept = np.zeros((params_train.shape[1],1))
    training_performances = []

    np.random.seed(1)

    for i in range(params_train.shape[1]):

        #Mdl[i] = sk.ensemble.BaggingRegressor(estimator=sklearn.tree.DecisionTreeRegressor(),n_estimators=n_trees,bootstrap=True,oob_score=True).fit(database_train,params_train[:,i])
        RFregress = BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=n_trees,bootstrap=True,oob_score=True)
        Mdl.append(RFregress.fit(database_train,params_train[:,i]))
        MLprediction[:,i] = Mdl[i].predict(database_train)
        
        training_performances.append(Mdl[i].oob_score_)

        linmodel = LinearRegression()
        X = np.transpose(params_train[:,i]).reshape(-1,1)
        y = np.transpose(MLprediction[:,i]).reshape(-1,1)
        linmodel.fit(X,y)

        Slope[i] = linmodel.coef_
        Intercept[i] = linmodel.intercept_

    trainedML = {'Mdl':Mdl,'training_performances':training_performances,'Slope':Slope,'Intercept':Intercept,}

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - RF trained in {round(tottime)} sec.')
    f.close()

    return trainedML