import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def train_MLP_python(database_train, params_train, n_layers, n_neurons, n_MLPs, method, log):

    # Train a Multi Layer Perceptron regressor for SANDI fitting

    # Author:
    # Dr. Marco Palombo
    # Cardiff University Brain Research Imaging Centre (CUBRIC)
    # Cardiff University, UK
    # 4th August 2022
    # Email: palombom@cardiff.ac.uk

    # Ported into python by Bradley Karat
    # University of Western Ontario
    # 20th June 2023
    # Email: bkarat@uwo.ca

    tic = time.time()

    method = method # If set to '0' it will create and train an MLP for each model parameter. If '1' it will create and train a single MLP for the prediciton of all the model parameters.

    net_structure = np.zeros((n_layers),dtype=int)

    for i in range(n_layers):
        net_structure[i] = int(n_neurons)

    np.random.seed(1)

    if method == 1:

        Mdl = []
        training_performances = []

        MLprediction = np.zeros((params_train.shape[0],params_train.shape[1], n_MLPs))
        Rsq = np.zeros((params_train.shape[1],n_MLPs))
        Slope = np.zeros((params_train.shape[1],n_MLPs))
        Intercept = np.zeros((params_train.shape[1],n_MLPs))

        for j in range(n_MLPs):
            
            MLPregress = MLPRegressor(net_structure,activation='relu',solver='adam')
            Mdl.append(MLPregress.fit(database_train,params_train))
            training_performances.append((MLPregress.score(database_train,params_train)))

            MLprediction[:,:,j] = Mdl[j].predict(database_train)

            for i in range(params_train.shape[1]):

                linmodel = LinearRegression()
                X = np.transpose(params_train[:,i]).reshape(-1,1)
                y = np.transpose(MLprediction[:,i]).reshape(-1,1)
                linmodel.fit(X,y)

                Slope[i,j] = linmodel.coef_
                Intercept[i,j] = linmodel.intercept_

            f = open(log, "a")
            f.write(f'\nMLP {j} / {n_MLPs} trained.')
            f.close()
    else:
        
        Mdl = np.full((params_train.shape[1],n_MLPs), 1).tolist() #previously we have been able to just do list.append in lieu of cells which works in the 1D case, however, we need a 2D list now to keep track of models
        training_performances = np.full((params_train.shape[1],n_MLPs), 1).tolist()

        MLprediction = np.zeros((params_train.shape[0],params_train.shape[1], n_MLPs))
        Rsq = np.zeros((params_train.shape[1],n_MLPs))
        Slope = np.zeros((params_train.shape[1],n_MLPs))
        Intercept = np.zeros((params_train.shape[1],n_MLPs))

        for j in range(n_MLPs):
            for i in range(params_train.shape[1]):

                f = open(log, "a")
                f.write(f'\nMLP for model parameter {i} / {params_train.shape[1]} training.')
                f.close()

                MLPregress = MLPRegressor(net_structure,activation='relu',solver='adam')
                Mdl[i][j] = MLPregress.fit(database_train,params_train[:,i])
                training_performances[i][j] = MLPregress.score(database_train,params_train[:,i])

                MLprediction[:,i,j] = Mdl[i][j].predict(database_train)
                
                linmodel = LinearRegression()
                X = np.transpose(params_train[:,i]).reshape(-1,1)
                y = np.transpose(MLprediction[:,i,j]).reshape(-1,1)
                linmodel.fit(X,y)

                Slope[i,j] = linmodel.coef_
                Intercept[i,j] = linmodel.intercept_

            f = open(log, "a")
            f.write(f'\nMLP {j} / {n_MLPs} trained.')
            f.close()


    trainedML = {'Mdl':Mdl,'training_performances':training_performances,'Slope':Slope,'Intercept':Intercept}

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - MLP trained in {round(tottime)} sec.')
    f.close()

    return trainedML


