import os
import glob
import sys

import pandas  as pd
import numpy   as np
import itea    as sr

from itertools               import product
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import KFold
    

def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())


# gridsearch_configurations is a dictionary, where each key is a parameter
# and its value can be one of two options:
# - list (python native):
#       contains the values that will be used in the search
# - lambda functions:
#       used for dynamic parameters, that assumes the value based on others

# Creation of parameters: a Cartesian product will be made
# over all passed lists, then the lambda functions will be applied
# about each configuration obtained.

gridsearch_configurations = {
    'popsize'   : [100, 250, 500],
    'gens'      : lambda conf:  100000//conf['popsize'],
    'minterms'  : [2],
    'model'     : [LinearRegression(n_jobs=-1)],
    'expolim'   : [(-3, 3), (-2, 2)],
    'maxterms'  : [10, 15],
    'check_fit' : [True],
    'funs'      : [{ # should be unary functions, f:R -> R
        'sin'      : np.sin,
        'cos'      : np.cos,
        'tan'      : np.tan,
        'abs'      : np.abs,
        'id'       : lambda x: x,
        'sqrt.abs' : lambda x: np.sqrt(np.absolute(x)),
        'log'      : np.log, 
        'exp'      : np.exp,
    }]
}

keys, values, varying = [], [], []
for k,v in gridsearch_configurations.items():
    if isinstance(v, list): 
        values.append(v)
        if len(v) > 1: # Salvando quem varia para printar dicionário colorido
            varying.append(k)
    elif callable(v): 
        continue
    else:
        raise Exception('o que é isso?')
    keys.append(k)
        
confs = [dict(zip(keys,items)) for items in product(*values)]

for conf in confs:
    for k,v in gridsearch_configurations.items():
        if callable(v):
            conf[k] = v(conf)
            varying.append(k)
                

# Saving the configuration informations
confs_df = pd.DataFrame(confs, index=[f'conf {i}' for i in range(len(confs))]).T
confs_df.index.names = ['Parameters']
confs_df.to_csv('itea-gridsearch_configurations.csv')


# function that takes a train and test dataset, the number of columns and generations
# and performs a single run.
def run(dataset_train, dataset_test, **params):
   
    Xtrain, ytrain = dataset_train[:, :-1], dataset_train[:, -1]
    Xtest,  ytest  = dataset_test[:, :-1],  dataset_test[:, -1]
    
    itea    = sr.ITEA(**params)
    bestsol = itea.run(Xtrain, ytrain, verbose=True)

    return RMSE(bestsol.predict(Xtrain).ravel(), ytrain.ravel()), RMSE(bestsol.predict(Xtest).ravel(), ytest.ravel()), str(bestsol)
    
    
# Our gridsearch implementation
def gridsearch(dataset_train, confs, ds, fold):
    
    # Creating checkpoints during the gridsearch
    gridDF = pd.DataFrame(columns = ['dataset', 'Fold', 'conf'] + [f'RMSE_{i}' for i in range(5)])
    gridDF = gridDF.set_index(['dataset', 'Fold', 'conf'])

    if os.path.isfile(f'../../results/gridsearch/itea-gridsearch.csv'):
        gridDF = pd.read_csv(f'../../results/gridsearch/itea-gridsearch.csv')
        gridDF = gridDF.set_index(['dataset', 'Fold', 'conf'])
    
    # Random state to make reproductible
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # (rmse_cv, configuration, configuration index)
    best_conf = (np.inf, {}, -1)
    
    for i, conf in enumerate(confs):
        if gridDF.index.isin([(ds, fold, i)]).any():
            print(f'successfully loaded result for configuration {i}')
        else:
            print(f'Testing configuration {i}/{len(confs)}', end='')
            # Vamos criar a linha e preencher com nan

            gridDF.loc[(ds, fold, i), :] = (np.nan, np.nan, np.nan, np.nan, np.nan)

            gridDF.to_csv(f'../../results/gridsearch/itea-gridsearch.csv', index=True)

        RMSE_cv = []
        for j, (train_index, test_index) in enumerate(kf.split(dataset_train)):
            if not np.isnan(gridDF.loc[(ds, fold, i), f'RMSE_{j}']):
                RMSE_cv.append(gridDF.loc[(ds, fold, i), f'RMSE_{j}'])
                print(f'recovered information for fold {j}: {RMSE_cv[-1]}')

                continue
            else:
                print(f'evauating fold {j} on cross-validation...')

            RMSE_train, RMSE_test, _ = run(dataset_train[train_index, :], dataset_train[test_index, :], **conf)
        
            RMSE_cv.append(RMSE_test)

            # Here we know that this line exists
            gridDF.loc[(ds, fold, i), f'RMSE_{j}'] = RMSE_test
            
            gridDF.to_csv(f'../../results/gridsearch/itea-gridsearch.csv', index=True)

        print(f': {np.mean(RMSE_cv)}, {RMSE_cv}')
        if np.mean(RMSE_cv) < best_conf[0]:
            best_conf = (np.mean(RMSE_cv), conf,  i)
            
    return best_conf


if __name__ == '__main__':        
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

    if len(sys.argv)== 1:
        print(f'Please pass as argument the data set to execute. Possible are: {datasets}')
        sys.exit()

    if str(sys.argv[1]) not in datasets:
        print(f'data set {str(sys.argv[1])} not found.')
        sys.exit()

    ds = str(sys.argv[1])

    columns   = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test','expr']

    datasets_folder = '../../../datasets/commaSeparated'
    fname     = f'../../results/rmse/itea-{ds}-resultsregression.csv'

    results   = {c:[] for c in columns}
    resultsDF = pd.DataFrame(columns=columns)

    if os.path.isfile(fname):
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')

    for fold in range(n_folds):
        print(f'Gridsearch --- data set: {ds}')
        dataset_train, dataset_test = None, None
        
        try:
            dataset_train = np.loadtxt(f'{datasets_folder}/{ds}-train-{fold}.dat', delimiter=',')
            dataset_test  = np.loadtxt(f'{datasets_folder}/{ds}-test-{fold}.dat', delimiter=',')
        except:
            print(f'Dataset {dataset_train} does not exist.')
            continue
            

        RMSE_cv, conf, conf_id = None, None, None
        if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
            aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]
            conf_id = aux_resultsDF['conf'].values[0]
            RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]
            conf    = confs[conf_id]

            print(f'Using previously configuration: {RMSE_cv}, {conf_id}')
        else:
            print('Evaluating fold in gridsearch') 
            RMSE_cv, conf, conf_id = gridsearch(dataset_train, confs, ds, fold)

        for rep in range(runs_per_fold):
            if len(resultsDF[
                (resultsDF['dataset']==ds) &
                (resultsDF['Fold']==fold)  &
                (resultsDF['Rep']==rep)
            ])==1:
                print(f'already evaluated {ds}-{fold}-{rep}')

                continue

            print(f'evaluating config {conf_id} for {ds}-{fold}-{rep}')
            
            RMSE_train, RMSE_test, expr = run(dataset_train, dataset_test, **conf)

            results['dataset'].append(ds)
            results['conf'].append(conf_id)
            results['RMSE_cv'].append(RMSE_cv)
            results['RMSE_train'].append(RMSE_train)
            results['RMSE_test'].append(RMSE_test)
            results['Fold'].append(fold)
            results['Rep'].append(rep)
            results['expr'].append(expr)

            df = pd.DataFrame(results)
            df.to_csv(fname, index=False)

    print('done')