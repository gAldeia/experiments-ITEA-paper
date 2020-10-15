import os
import glob
import sys
import warnings
import time

import pandas  as pd
import numpy   as np

from itertools               import product
from sklearn.model_selection import KFold
from gplearn.genetic         import SymbolicRegressor
from gplearn.functions       import make_function


warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
np.seterr(all='ignore')


# Creating functions as they are implemented on the ITEA
def plog(x):
    z = np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.0)

    z = np.where(np.abs(z) < 1e+1000, z, np.sign(z)*1e+1000)
    z[~np.isfinite(z)]=0

    return z

def pdiv(x,y):
    z = np.where(np.abs(y) > 0.01, x/y, 1.0)

    z = np.where(np.abs(z) < 1e+1000, z, np.sign(z)*1e+1000)
    z[~np.isfinite(z)]=0

    return z

def ptan(x):
    z = np.tanh(x)

    z = np.where(np.abs(z) < 1e+1000, z, np.sign(z)*1e+1000)
    z[~np.isfinite(z)]=0
    
    return z

def pexp(x):
    z = np.where(np.exp(x) <= np.exp(100), np.exp(x), np.exp(100))
    
    z = np.where(np.abs(z) < 1e+1000, z, np.sign(z)*1e+1000)
    z[~np.isfinite(z)]=0
    
    return z

myExp  = make_function(pexp, "exp", 1)
#myTanh = make_function(np.tanh, "tanh", 1)
mySqrt = make_function(lambda x: np.sqrt(np.abs(x)), "sqrtabs", 1)
myLog  = make_function(plog, "log",1)
myDiv  = make_function(pdiv, "pdiv", 2)
myTanh = make_function(ptan, "tan", 1)
        
    

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
    'population_size'     : [100, 250, 500],
    'generations'         : lambda conf: 100000//conf['population_size'],
    'p_crossover'         : [0.2, 0.5, 0.8],
    'p_subtree_mutation'  : lambda conf: 0.2*(1 - conf['p_crossover'])/5,
    'p_point_mutation'    : lambda conf: 0.2*(1 - conf['p_crossover'])/5,
    'p_hoist_mutation'    : lambda conf: 0.1*(1 - conf['p_crossover'])/5,
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
confs_df.to_csv('gplearn-gridsearch_configurations.csv')


# function that takes a train and test dataset, the parameters to use in the
# algorithm, and performs a single run.
def run(dataset_train, dataset_test,
        population_size, generations, p_crossover, p_subtree_mutation, p_point_mutation, p_hoist_mutation):

    Xtrain, ytrain = dataset_train[:, :-1], dataset_train[:, -1]
    Xtest,  ytest  = dataset_test[:, :-1],  dataset_test[:, -1]
    
    f_set = ('add', 'sub', 'mul', myDiv, 'sin', 'cos', myLog, mySqrt, myTanh, myExp) 

    est_gp = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        stopping_criteria=0.01,
        p_crossover=p_crossover,
        p_subtree_mutation=p_subtree_mutation,
        p_hoist_mutation=p_hoist_mutation,
        p_point_mutation=p_point_mutation,
        max_samples=1.0,
        verbose=0,
        parsimony_coefficient=0.05,
        function_set = f_set,
        n_jobs=1
    )
    
    est_gp.fit(Xtrain, ytrain)
    
    return RMSE(est_gp.predict(Xtrain), ytrain), RMSE(est_gp.predict(Xtest), ytest)
    
    
# Our gridsearch implementation
def gridsearch(dataset_train, confs, ds, fold):
    
    # Creating checkpoints during the gridsearch
    gridDF = pd.DataFrame(columns = ['dataset', 'Fold', 'conf'] + [f'RMSE_{i}' for i in range(5)])
    gridDF = gridDF.set_index(['dataset', 'Fold', 'conf'])

    if os.path.isfile(f'../../results/gridsearch/gplearn-{ds}-gridsearch.csv'):
        gridDF = pd.read_csv(f'../../results/gridsearch/gplearn-{ds}-gridsearch.csv')
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

            gridDF.loc[(ds, fold, i), :] = (np.nan, np.nan, np.nan, np.nan, np.nan)

            gridDF.to_csv(f'../../results/gridsearch/gplearn-{ds}-gridsearch.csv', index=True)

        RMSE_cv = []
        for j, (train_index, test_index) in enumerate(kf.split(dataset_train)):
            if not np.isnan(gridDF.loc[(ds, fold, i), f'RMSE_{j}']):
                RMSE_cv.append(gridDF.loc[(ds, fold, i), f'RMSE_{j}'])
                print(f'recovered information for fold {j}: {RMSE_cv[-1]}')

                continue
            else:
                print(f'evauating fold {j} on cross-validation...')

            # Remaining time estimation
            t = time.time()

            RMSE_train, RMSE_test = run(dataset_train[train_index, :], dataset_train[test_index, :], **conf)
        
            last_t = time.time() - t

            print(f'Remaining time: {get_remaining_time(last_t)}')
            
            RMSE_cv.append(RMSE_test)

            # Here we know that this line exists
            gridDF.loc[(ds, fold, i), f'RMSE_{j}'] = RMSE_test
            
            gridDF.to_csv(f'../../results/gridsearch/gplearn-{ds}-gridsearch.csv'', index=True)

        print(f': {np.mean(RMSE_cv)}, {RMSE_cv}')
        if np.mean(RMSE_cv) < best_conf[0]:
            best_conf = (np.mean(RMSE_cv), conf,  i)
            
    return best_conf


# Estimating remaining time:
# (source: https://stackoverflow.com/questions/44926127/calculating-the-amount-of-time-left-until-completion)
last_times        = []
counter           = 0
n_datasets        = 10
n_confs           = len(confs)
n_rep_per_dataset = 30
n_kfold           = 5
def get_remaining_time(time):
    global last_times        
    global counter           
    global n_datasets        
    global n_confs           
    global n_rep_per_dataset 
    global n_kfold      

    last_times.append(time)
    len_last_t = len(last_times)

    if len_last_t > 5:
        last_times.pop(0)

    mean_t = sum(last_times) // len_last_t 
    remain_s_tot = mean_t * (n_datasets * n_confs * n_rep_per_dataset * n_kfold - counter + 1) 
    remain_m = remain_s_tot // 60
    remain_s = remain_s_tot % 60
    counter += 1

    return f"{remain_m}m{remain_s}s"


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

    columns   = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

    datasets_folder = '../../datasets/commaSeparated'
    fname = f'../../results/rmse/gplearn-{ds}-resultsregression.csv'

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

        # We will check if the fold was already evaluated in a previous repetition,
        # and use this result for the remaining repetitions ON THE SAME FOLD (notice that there
        # are 5 folds).
        RMSE_cv, conf, conf_id = None, None, None
        if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
            aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]
            conf_id = aux_resultsDF['conf'].values[0]
            RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]
            conf    = confs[conf_id]

            print(f'Using previously configuration: {RMSE_cv}, {conf_id}')
        else:
            # There is no repetition on this specific fold, so we need to
            # do the first.
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
            
            RMSE_train, RMSE_test = run(dataset_train, dataset_test, **conf)

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