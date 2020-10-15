import os

import pandas  as pd
import numpy   as np

from sklearn.metrics         import mean_squared_error, make_scorer
from sklearn.tree            import DecisionTreeRegressor
from itertools               import product
from sklearn.model_selection import GridSearchCV


def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())
    
# Tree is not deterministic --- running multiple times for each fold
n_folds       = 5
n_runs        = 30
runs_per_fold = n_runs//n_folds

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

# In this case, we are saving the RMSE on the cross validation, and the number
# of the repetition.
columns = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

datasets_folder = '../../datasets/commaSeparated'
fname = '../../results/rmse/tree-resultsregression.csv'

results = {c:[] for c in columns}
resultsDF = pd.DataFrame(columns = columns)

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
            
        print(f'Executing fold {fold}')
            
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test,  y_test  = dataset_test[:, :-1],  dataset_test[:, -1]

        # Dinamically changing the min_samples_split according to the data set
        confs = {
            'criterion' : ['mse'],
            'min_samples_split' : [len(X_train)//100, len(X_train)*5//100, len(X_train)*10//100],
        }
        
        # We will check if the fold was already evaluated in a previous repetition,
        # and use this result for the remaining repetitions ON THE SAME FOLD (notice that there
        # are 5 folds).
        RMSE_cv, conf, conf_id = None, None, None
        if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
            aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]
            conf_id       = aux_resultsDF['conf'].values[0]
            RMSE_cv       = aux_resultsDF['RMSE_cv'].values[0]

            # We will recover the information and use for the remaining repetitions
            keys, values = [], []
            for k,v in confs.items():
                if isinstance(v, list): 
                    values.append(v)
                elif callable(v): 
                    continue
                else:
                    raise Exception('Error when recovering informations')
                keys.append(k)
                    
            conf = [dict(zip(keys,items)) for items in product(*values)][conf_id]

            print(f'Using previously configuration: {RMSE_cv}, {conf_id}')
        else:
            # There is no repetition on this specific fold, so we need to
            # do the first.
            print('Evaluating fold in gridsearch')
            
            # Using 5 folds and cross validation. We need to convert our RMSE function to be compatible
            # with the gridsearch, through the make_scorer. To make it a minimization problem, the
            # greater_is_better will make the RMSE a negative value (so it is analogous to a maximization
            # problem). We need to change this later on the results file.
            grid = GridSearchCV(
                DecisionTreeRegressor(),
                confs,
                n_jobs=4,
                cv=5,
                verbose=1,
                scoring=make_scorer(RMSE, greater_is_better=False),
                return_train_score=True
            ).fit(X_train, y_train)

            conf    = grid.best_params_
            RMSE_cv = -1*grid.best_score_

            # Saving the id to recover if needed
            conf_id = np.where(grid.cv_results_['rank_test_score'] ==1)[0][0]

        # Once we have the best configuration in the gridsearch, we will use it
        # on the repetitions over the same fold
        for rep in range(runs_per_fold):

            # Returning from the last checkpoint
            if len(resultsDF[
                (resultsDF['dataset']==ds) &
                (resultsDF['Fold']==fold)  &
                (resultsDF['Rep']==rep)
            ])==1:
                print(f'already evaluated {ds}-{fold}-{rep}')
                continue

            print(f'evaluating config {conf_id} for {ds}-{fold}-{rep}')
            
            regressor = DecisionTreeRegressor(**grid.best_params_).fit(X_train, y_train)  

            RMSE_train = RMSE(regressor.predict(X_train).ravel(), y_train.ravel())
            RMSE_test  = RMSE(regressor.predict(X_test).ravel(), y_test.ravel())

            results['dataset'].append(ds)
            results['conf'].append(conf_id)
            results['RMSE_cv'].append(RMSE_cv)
            results['RMSE_train'].append(RMSE_train)
            results['RMSE_test'].append(RMSE_test)
            results['Fold'].append(fold)
            results['Rep'].append(rep)

            df = pd.DataFrame(results)
            df.to_csv(fname, index=False)

print('done')
