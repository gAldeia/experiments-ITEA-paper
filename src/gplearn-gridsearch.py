import os
import glob
import sys

import os.path as path
import pandas  as pd
import numpy   as np
import time

from itertools         import product
from sklearn.metrics   import mean_squared_error, mean_absolute_error, make_scorer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

datasets_folder = '../datasets/'


# ------------------------------------------------------------------
# Código do Fabrício - tudo menos o main
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function


# ocultar warnings
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

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

myExp = make_function(pexp, "exp", 1)
#myTanh = make_function(np.tanh, "tanh", 1)
mySqrt = make_function(lambda x: np.sqrt(np.abs(x)), "sqrtabs", 1)
myLog = make_function(plog, "log",1)
myDiv = make_function(pdiv, "pdiv", 2)
myTanh = make_function(ptan, "tan", 1)
        
    
def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())


# Criação das configurações

# gridsearch_configuration é um dicionário, onde cada key é um parâmetro
# e o seu valor pode ser uma das duas opções:
# - lista (nativa): contém os valores que serão utilizados na busca (colocar só 1 se for fixar)
# - funções lambda: usada para parâmetros que assumem o valor baseado em outros

# Criação dos parâmetros: será feito um produto cartesiano
# sobre todas as listas passadas, e então as funções lambda serão aplicadas
# sobre cada configuração obtida pelo prod. cartesiano.

gridsearch_configurations = {
    'population_size'        : [100, 250, 500],
    'generations'         : lambda conf: 100000//conf['population_size'],
    'p_crossover'            : [0.2, 0.5, 0.8],
    'p_subtree_mutation'     : lambda conf: 0.2*(1 - conf['p_crossover'])/5,
    'p_point_mutation'       : lambda conf: 0.2*(1 - conf['p_crossover'])/5,
    'p_hoist_mutation'       : lambda conf: 0.1*(1 - conf['p_crossover'])/5,
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

# Criando um dataframe para enumerar e visualizar melhor as configurações
confs_df = pd.DataFrame(confs, index=[f'conf {i}' for i in range(len(confs))]).T
confs_df.index.names = ['Parameters']
confs_df.to_csv('gplearn-gridsearch_configurations.csv')


# Função que recebe um dataset e uma configuração e executa o algoritmo
def run(dataset_train, dataset_test, population_size, generations, p_crossover, p_subtree_mutation, p_point_mutation, p_hoist_mutation):

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
    
    
# Função que faz a busca pela melhor configuração:
def gridsearch(dataset_train, confs, ds, fold):
    
    # Para salvar o andamento do gridsearch e poder retomar
    gridDF = pd.DataFrame(columns = ['dataset', 'Fold', 'conf'] + [f'RMSE_{i}' for i in range(5)])
    gridDF = gridDF.set_index(['dataset', 'Fold', 'conf'])

    if os.path.isfile(f'../docs/gplearn-{ds}-gridsearch.csv'):
        gridDF = pd.read_csv(f'../docs/gplearn-{ds}-gridsearch.csv')
        gridDF = gridDF.set_index(['dataset', 'Fold', 'conf'])
    
    # Random state para poder retomar
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # (rmse_cv, configuração, indice da configuração)
    best_conf = (np.inf, {}, -1)
    
    for i, conf in enumerate(confs):
        # Verificando se esse fold de cv já foi avaliado.
        # A linha da configuração foi criada - vamos pegar os valores calculados ou calcular um novo
        if gridDF.index.isin([(ds, fold, i)]).any():
            print(f'Retomando gridsearch para configuração {i}')
        else:
            print(f'Testando configuração {i}/{len(confs)}')
            # Vamos criar a linha e preencher com nan

            gridDF.loc[(ds, fold, i), :] = (np.nan, np.nan, np.nan, np.nan, np.nan)

            gridDF.to_csv(f'../docs/gplearn-{ds}-gridsearch.csv', index=True)

        RMSE_cv = []
        for j, (train_index, test_index) in enumerate(kf.split(dataset_train)):
            if not np.isnan(gridDF.loc[(ds, fold, i), f'RMSE_{j}']):
                RMSE_cv.append(gridDF.loc[(ds, fold, i), f'RMSE_{j}'])
                print(f'Recuperando RMSE do fold de cv {j}: {RMSE_cv[-1]}')

                continue
            else:
                print(f'Calculando RMSE do fold de cv {j}...')

            # Estimar tempo restante
            t = time.time()

            RMSE_train, RMSE_test = run(dataset_train[train_index, :], dataset_train[test_index, :], **conf)
        
            # Estimar tempo restante
            last_t = time.time() - t

            print(f'Tempo restante estimado: {get_remaining_time(last_t)}')
            
            RMSE_cv.append(RMSE_test)

            # Aqui sabemos que a linha existe pois ou já existia ou foi criada
            gridDF.loc[(ds, fold, i), f'RMSE_{j}'] = RMSE_test
            
            # Salvar essa nova avaliação
            gridDF.to_csv(f'../docs/gplearn-{ds}-gridsearch.csv', index=True)

        print(f': {np.mean(RMSE_cv)}, {RMSE_cv}')
        if np.mean(RMSE_cv) < best_conf[0]:
            best_conf = (np.mean(RMSE_cv), conf,  i)
            
    return best_conf


# Função para imprimir estimativa de tempo:
# Fonte: https://stackoverflow.com/questions/44926127/calculating-the-amount-of-time-left-until-completion
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
    # Gridsearch

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
        print(f'Informe um dataset para executar. Possiveís datasets: {datasets}')
        sys.exit()

    if str(sys.argv[1]) not in datasets:
        print(f'dataset {str(sys.argv[1])} não conhecido.')
        sys.exit()

    # Pegando dataset passado
    ds = str(sys.argv[1]) #lista de argumentos, índice 0 é o nome do programa. 

    # ---------------------------
    columns   = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

    fname     = f'../docs/gplearn-{ds}-resultsregression.csv'

    results   = {c:[] for c in columns}
    resultsDF = pd.DataFrame(columns=columns)

    if os.path.isfile(fname):
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')

    print(f'Executando agora para o dataset {ds}')
    for fold in range(n_folds):
        dataset_train, dataset_test = None, None
        
        # evitar tentar abrir arquivos que não existem
        try:
            dataset_train = np.loadtxt(f'{datasets_folder}/{ds}-train-{fold}.dat', delimiter=',')
            dataset_test  = np.loadtxt(f'{datasets_folder}/{ds}-test-{fold}.dat', delimiter=',')
        except:
            print(f'Dataset {dataset_train} does not exist.')
            continue
            
        print(f'Executando para o fold {fold}')

        RMSE_cv, conf, conf_id = None, None, None
        if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
            # Verificar se aquele fold já foi avaliado em alguma repetição, e caso tenha sido
            # pega a configuração utilizada em uma delas (vao ser todas iguais, tanto faz a repetição
            # contanto que seja no mesmo fold)
            aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]
            conf_id = aux_resultsDF['conf'].values[0]
            RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]
            conf    = confs[conf_id]

            print(f'Pegando configuração avaliada anteriormente: {RMSE_cv}, {conf_id}')
        else:
            # Obtendo melhor configuração para esse treino-teste
            print('Obtendo a melhor configuração utilizando 5-fold cv no treino')
            RMSE_cv, conf, conf_id = gridsearch(dataset_train, confs, ds, fold)

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
            results['conf'].append(conf_id)
            results['RMSE_cv'].append(RMSE_cv)
            results['RMSE_train'].append(RMSE_train)
            results['RMSE_test'].append(RMSE_test)
            results['Fold'].append(fold)
            results['Rep'].append(rep)

            df = pd.DataFrame(results)
            df.to_csv(fname, index=False)

    print('done')