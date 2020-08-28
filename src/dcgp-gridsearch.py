import os
import glob

import os.path as path
import pandas  as pd
import numpy   as np

from itertools         import product
from sklearn.metrics   import mean_squared_error, mean_absolute_error, make_scorer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

datasets_folder = '../datasets/'


# ----------------------------------------------------------------------------------------------
import dcgpy
import pygmo as pg
print(dcgpy.__version__)
    
    
def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())


# ----------------------------------------------------------------------------------------------
# Criação das configurações

# gridsearch_configuration é um dicionário, onde cada key é um parâmetro
# e o seu valor pode ser uma das duas opções:
# - lista (nativa): contém os valores que serão utilizados na busca (colocar só 1 se for fixar)
# - funções lambda: usada para parâmetros que assumem o valor baseado em outros

# Criação dos parâmetros: será feito um produto cartesiano
# sobre todas as listas passadas, e então as funções lambda serão aplicadas
# sobre cada configuração obtida pelo prod. cartesiano.

gridsearch_configurations = {
    'cols' : [100, 250, 500],
    'gen'  : lambda conf:  100000//conf['cols']
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
                
# Verificar se temos em todas as confs o mesmo número
# que deveriamos ter do dicionário de valores
for conf in confs:
    if set(conf.keys()) != set(gridsearch_configurations.keys()):
        raise Exception(f'Parâmetros de busca e da configuração específica não batem. Configuração:\n{conf}')

# Criando um dataframe para enumerar e visualizar melhor as configurações
confs_df = pd.DataFrame(confs, index=[f'conf {i}' for i in range(len(confs))]).T
confs_df.index.names = ['Parameters']
confs_df.to_csv('dcgp-gridsearch_configurations.csv')


# ----------------------------------------------------------------------------------------------
# Função que recebe um dataset e uma configuração e executa o algoritmo
def run(dataset_train, dataset_test, cols, gen):
    # kernels: https://darioizzo.github.io/dcgp/docs/cpp/kernel_list.html
    ss = dcgpy.kernel_set_double(["sum", "diff", "mul", "pdiv", "sin", "cos", "tanh", "log", "exp", "psqrt"])
        
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
    
    
# Função que faz a busca pela melhor configuração:
def gridsearch(dataset_train, confs):
    
    kf = KFold(n_splits=5, shuffle=True)

    # (rmse_cv, configuração, indice da configuração)
    best_conf = (np.inf, {}, -1)
    
    for i, conf in enumerate(confs):
        print(f'Testando configuração {i}/{len(confs)}', end='')
        
        RMSE_cv = []
        for train_index, test_index in kf.split(dataset_train):
            RMSE_train, RMSE_test = run(dataset_train[train_index, :], dataset_train[test_index, :], **conf)
            RMSE_cv.append(RMSE_test)

        print(f': {np.mean(RMSE_cv)}, {RMSE_cv}')
        if np.mean(RMSE_cv) < best_conf[0]:
            best_conf = (np.mean(RMSE_cv), conf,  i)
            
    return best_conf


# ----------------------------------------------------------------------------------------------
if __name__=='__main__':
    # Gridsearch

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

    # ---------------------------
    columns   = ['dataset','conf_id','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

    fname     = '../docs/dCartesian-resultsregression.csv'

    results   = {c:[] for c in columns}
    resultsDF = pd.DataFrame(columns=columns)

    if os.path.isfile(fname):
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')

    for ds in datasets:
        print(f'Executando agora para o dataset {ds}')
        for fold in range(n_folds):
            dataset_train, dataset_test = None, None
            
            # evitar tentar abrir arquivos que não existem
            try:
                dataset_train = np.loadtxt(f'{datasets_folder}/{ds}-train-{fold}.dat', delimiter=',')
                dataset_test  = np.loadtxt(f'{datasets_folder}/{ds}-test-{fold}.dat', delimiter=',')
            except:
                print(f'Dataset {dataset_train} or {dataset_test} does not exist.')
                continue
                
            print(f'Executando para o fold {fold}')

            RMSE_cv, conf, conf_id = None, None, None
            if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
                # Verificar se aquele fold já foi avaliado em alguma repetição, e caso tenha sido
                # pega a configuração utilizada em uma delas (vao ser todas iguais, tanto faz a repetição
                # contanto que seja no mesmo fold)
                aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]

                conf_id = aux_resultsDF['conf_id'].values[0]
                RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]
                conf    = confs[conf_id]

                print(f'Pegando configuração avaliada anteriormente: {RMSE_cv}, {conf_id}')
            else:
                # Obtendo melhor configuração para esse treino-teste
                print('Obtendo a melhor configuração utilizando 5-fold cv no treino')
                RMSE_cv, conf, conf_id = gridsearch(dataset_train, confs)

            print('Começando a testar a melhor configuração obtida')
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
    