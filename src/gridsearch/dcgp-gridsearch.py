import os
import glob
import dcgpy

import pandas  as pd
import numpy   as np
import pygmo   as pg

from itertools               import product
from sklearn.model_selection import KFold

# Documentation: https://darioizzo.github.io/dcgp/installation.html
# Kernels:       https://darioizzo.github.io/dcgp/docs/cpp/kernel_list.html
    
    
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
    'cols' : [100, 250, 500],
    'gen'  : lambda conf:  100000//conf['cols']
}

keys, values, varying = [], [], []
for k,v in gridsearch_configurations.items():
    if isinstance(v, list): 
        values.append(v)
        if len(v) > 1:
            varying.append(k)
    elif callable(v): 
        continue
    else:
        raise Exception('Error creating the configurations')
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
confs_df.to_csv('dcgp-gridsearch_configurations.csv')


# function that takes a train and test dataset, the number of columns and generations
# and performs a single run.
def run(dataset_train, dataset_test, cols, gen):
    ss = dcgpy.kernel_set_double(
        ["sum", "diff", "mul", "pdiv", "sin", "cos", "tanh", "log", "exp", "psqrt"]
    )
        
    Xtrain, ytrain = dataset_train[:, :-1], dataset_train[:, -1]
    Xtest,  ytest  = dataset_test[:, :-1],  dataset_test[:, -1]
    
    udp = dcgpy.symbolic_regression(
        points = Xtrain, labels = ytrain[:,np.newaxis], kernels=ss(), 
        rows=1, cols=cols, levels_back=21, arity=2, 
        n_eph=3, multi_objective=False, parallel_batches = 0
    )
    
    uda  = dcgpy.es4cgp(gen = gen) #, mut_n = 1)

    algo = pg.algorithm(uda)
    pop = pg.population(udp, 4)
    pop = algo.evolve(pop)     
    
    return RMSE(udp.predict(Xtrain, pop.champion_x), ytrain), RMSE(udp.predict(Xtest, pop.champion_x), ytest)
    
    
# Our gridsearch implementation
def gridsearch(dataset_train, confs):
    
    kf = KFold(n_splits=5, shuffle=True)

    # (rmse_cv, configuration, configuration index)
    best_conf = (np.inf, {}, -1)
    
    for i, conf in enumerate(confs):
        print(f'Testing configuration {i}/{len(confs)}', end='')
        
        RMSE_cv = []
        for train_index, test_index in kf.split(dataset_train):
            RMSE_train, RMSE_test = run(dataset_train[train_index, :], dataset_train[test_index, :], **conf)
            RMSE_cv.append(RMSE_test)

        print(f': {np.mean(RMSE_cv)}, {RMSE_cv}')
        if np.mean(RMSE_cv) < best_conf[0]:
            best_conf = (np.mean(RMSE_cv), conf,  i)
            
    return best_conf


if __name__=='__main__':
    n_folds       = 5
    n_runs        = 30
    runs_per_fold = n_runs//n_folds

    datasets = [
        'airfoil',
        'concrete',
        'energyCooling',
        'energyHeating',
        'GeographicalOriginalofMusic',
        'towerData',
        'tecator',
        'wineRed',
        'wineWhite',
        'yacht',
    ]    

    columns   = ['dataset','conf_id','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

    datasets_folder = '../../datasets/commaSeparated'
    fname = '../../results/rmse/DCGP-resultsregression.csv'

    results   = {c:[] for c in columns}
    resultsDF = pd.DataFrame(columns=columns)

    if os.path.isfile(fname):
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')

    for ds in datasets:
        print(f'Gridsearch --- data set: {ds}')

        for fold in range(n_folds):
            dataset_train, dataset_test = None, None
            
            try:
                dataset_train = np.loadtxt(f'{datasets_folder}/{ds}-train-{fold}.dat', delimiter=',')
                dataset_test  = np.loadtxt(f'{datasets_folder}/{ds}-test-{fold}.dat', delimiter=',')
            except:
                print(f'Dataset {dataset_train} or {dataset_test} does not exist.')
                continue
                
            # We will check if the fold was already evaluated in a previous repetition,
            # and use this result for the remaining repetitions ON THE SAME FOLD (notice that there
            # are 5 folds).
            RMSE_cv, conf, conf_id = None, None, None
            if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
                aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]

                conf_id = aux_resultsDF['conf_id'].values[0]
                RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]
                conf    = confs[conf_id]

                print(f'Using previously configuration: {RMSE_cv}, {conf_id}')
            else:
                # There is no repetition on this specific fold, so we need to
                # do the first.
                print('Evaluating fold in gridsearch')
                
                RMSE_cv, conf, conf_id = gridsearch(dataset_train, confs)

            # Once we have the best configuration in the gridsearch, we will use it
            # on the repetitions over the same fold
            for rep in range(runs_per_fold):
                if len(resultsDF[
                    (resultsDF['dataset']==ds) &
                    (resultsDF['Fold']==fold)  &
                    (resultsDF['Rep']==rep)
                ])==1:
                    print(f'already evaluated {ds}-{fold}-{rep}')

                    continue

                print(f'evaluating config {conf_id} for {ds}-{fold}-{rep}')
                
                RMSE_train, RMSE_test = run(dataset_train, dataset_test, **conf)

                results['dataset'].append(ds)
                results['conf_id'].append(conf_id)
                results['RMSE_cv'].append(RMSE_cv)
                results['RMSE_train'].append(RMSE_train)
                results['RMSE_test'].append(RMSE_test)
                results['Fold'].append(fold)
                results['Rep'].append(rep)

                df = pd.DataFrame(results)
                df.to_csv(fname, index=False)

    print('done')