import os
import glob
import sys
import time

import pandas  as pd
import numpy   as np

from itertools               import product
from sklearn.model_selection import KFold
from feat                    import Feat


# Function to save the expression in a similar way to the ITEA expressions
def model2str(expr):
    e = []
    for t in expr.split('\n')[1:]:
        ts = t.split('\t\t')
        if len(ts) < 2:
           continue
        ti = ts[0]
        c =  ts[1]
        e.append(c + "*(" + ti + ")")
    return " + ".join(e)



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
    'pop_size'   : [100, 250, 500],
    'gens'       : lambda conf: 100000//conf['pop_size'],
    'cross_rate' : [0.2, 0.5, 0.8],
    'fb'         : [0.2, 0.5, 0.8]
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
confs_df.to_csv('feat-gridsearch_configurations.csv')


# function that takes a train and test dataset, the number of columns and generations
# and performs a single run.
def run(dataset_train, dataset_test, pop_size, gens, cross_rate, fb, max_time=1200):

    Xtrain, ytrain = dataset_train[:, :-1], dataset_train[:, -1]
    Xtest,  ytest  = dataset_test[:, :-1],  dataset_test[:, -1]
    
    est_gp = Feat(obj="fitness,complexity",
               pop_size=pop_size,
               gens=gens,
               max_time=max_time,
               max_stall=50,
               batch_size=10000,
               ml = "LinearRidgeRegression",
               sel='lexicase',
               surv='nsga2',
               max_depth=10,
               max_dim=min([Xtrain.shape[1]*2,50]),
               #random_state=random_seed,
               functions="+,-,*,/,sqrt,sin,cos,tanh,exp,log,^,x,kd",
               otype="f",
               backprop=True,
               iters=10,
               n_threads=1,
               verbosity=1,
               # tuned parameters
               cross_rate= cross_rate,
               fb = fb,
               root_xo_rate = 0.75,
               softmax_norm = False
               )
    
    est_gp.fit(Xtrain, ytrain)
    
    return RMSE(est_gp.predict(Xtrain), ytrain), RMSE(est_gp.predict(Xtest), ytest), est_gp.get_model()
    
    
# Our gridsearch implementation
def gridsearch(dataset_train, confs, ds, fold):

    # Creating checkpoints during the gridsearch
    gridDF = pd.DataFrame(columns = ['dataset', 'Fold', 'conf'] + [f'RMSE_{i}' for i in range(5)])
    gridDF = gridDF.set_index(['dataset', 'Fold', 'conf'])

    if os.path.isfile('../../results/gridsearch/feat-gridsearch.csv'):
        gridDF = pd.read_csv('../../results/gridsearch/feat-gridsearch.csv')
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

            gridDF.to_csv('../../results/gridsearch/feat-gridsearch.csv', index=True)

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

            RMSE_train, RMSE_test,_ = run(dataset_train[train_index, :], dataset_train[test_index, :], **conf, max_time=60)
        
            # Estimar tempo restante
            last_t = time.time() - t
            print(f'Remaining time: {get_remaining_time(last_t)}')
            
            RMSE_cv.append(RMSE_test)

            # Here we know that this line exists
            gridDF.loc[(ds, fold, i), f'RMSE_{j}'] = RMSE_test
            
            gridDF.to_csv('../../results/gridsearch/feat-gridsearch.csv', index=True)

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
        'GeographicalOriginalofMusic',
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

    columns   = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test', 'Expression']

    datasets_folder = '../../datasets/commaSeparated'
    fname     = f'../../results/rmse/FEAT-{ds}-resultsregression.csv'

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
            
            RMSE_train, RMSE_test, model = run(dataset_train, dataset_test, **conf)

            results['dataset'].append(ds)
            results['conf'].append(conf_id)
            results['RMSE_cv'].append(RMSE_cv)
            results['RMSE_train'].append(RMSE_train)
            results['RMSE_test'].append(RMSE_test)
            results['Fold'].append(fold)
            results['Rep'].append(rep)
            results['Expression'].append(model2str(model))

            df = pd.DataFrame(results)
            df.to_csv(fname, index=False)

    print('done')