import os

import pandas  as pd
import numpy   as np

from sklearn.metrics         import make_scorer
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())
    

# kNN is deterministic --- it doesn't need repetitions for each fold
n_folds         = 5
max_n_neighbors = 30 # represents x% of the data set

datasets = [
    'airfoil',
    'concrete',
    'energyCooling',
    'energyHeating',
    'Geographical',
    'towerData',
    'tecator',
    'wineRed',
    'wineWhite',
    'yacht',
]    

columns = ['dataset','conf','Fold','RMSE_train','RMSE_test']

datasets_folder = '../../datasets/commaSeparated'
fname = '../../results/rmse/knn-resultsregression.csv'

results = {c:[] for c in columns}

if os.path.isfile(fname):
    resultsDF = pd.read_csv(fname)
    results   = resultsDF.to_dict('list')

for ds in datasets:
    print(f'Gridsearch --- data set: {ds}')
    
    for fold in range(n_folds):
        
        dataset_train = None
        dataset_test  = None
        
        try:
            dataset_train = np.loadtxt(f'{datasets_folder}/{ds}-train-{fold}.dat', delimiter=',')
            dataset_test  = np.loadtxt(f'{datasets_folder}/{ds}-test-{fold}.dat', delimiter=',')
        except:
            continue
            
        # skip until reach the last checkpoint
        if os.path.isfile(fname):
            resultsDF = pd.read_csv(fname)
            results   = resultsDF.to_dict('list')

            if len( resultsDF[(resultsDF['dataset']==ds) & (resultsDF['Fold']==fold)] )==1:
                print(f'already evaluated {ds}-{fold}')
                continue
            
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test,  y_test  = dataset_test[:, :-1],  dataset_test[:, -1]
        
        # Calculating the number of observations, and using the specified percentage
        # as the maximum value
        confs = {
            'n_jobs'      : [None],
            'weights'     : ['distance', 'uniform'],
            'n_neighbors' : list(range(1, (len(X_train)*max_n_neighbors)//100))
        }

        print(f'evaluating {ds}-{fold}')
        
        # Using 5 folds and cross validation. We need to convert our RMSE function to be compatible
        # with the gridsearch, through the make_scorer. To make it a minimization problem, the
        # greater_is_better will make the RMSE a negative value (so it is analogous to a maximization
        # problem). We need to change this later on the results file.
        grid = GridSearchCV(
            KNeighborsRegressor(),
            confs,
            n_jobs=4,
            cv=5,
            verbose=1,
            scoring=make_scorer(RMSE, greater_is_better=False),
            return_train_score=True
        ).fit(X_train, y_train)

        # Using the best gridsearch configuration to train and obtain the final RMSEs
        regressor = KNeighborsRegressor(**grid.best_params_).fit(X_train, y_train)  

        RMSE_train = -1*grid.best_score_ 
        RMSE_test  = RMSE(regressor.predict(X_test).ravel(), y_test.ravel())

        results['dataset'].append(ds)
        results['conf'].append(grid.best_params_)
        results['RMSE_train'].append(RMSE_train)
        results['RMSE_test'].append(RMSE_test)
        results['Fold'].append(fold)

        df = pd.DataFrame(results)
        df.to_csv(fname, index=False)

print('done')