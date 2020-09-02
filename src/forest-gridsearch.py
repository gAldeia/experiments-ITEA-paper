import os

import os.path as path
import pandas  as pd
import numpy   as np

from sklearn.metrics   import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor

from itertools         import product
from sklearn.model_selection import GridSearchCV

datasets_folder = '../datasets/'


# Função de erro para o gridsearch 
def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())
    
n_folds         = 5
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

# ---------------------------
columns = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

fname = '../docs/forest-resultsregression.csv'
results = {c:[] for c in columns}
resultsDF = pd.DataFrame(columns = columns)

if os.path.isfile(fname):
    resultsDF = pd.read_csv(fname)
    results   = resultsDF.to_dict('list')

for ds in datasets:
    print(f'Gridsearch para base {ds}')
    
    for fold in range(n_folds):
        
        dataset_train = None
        dataset_test  = None
        
        # evitar tentar abrir arquivos que não existem
        try:
            dataset_train = np.loadtxt(f'{datasets_folder}/{ds}-train-{fold}.dat', delimiter=',')
            dataset_test  = np.loadtxt(f'{datasets_folder}/{ds}-test-{fold}.dat', delimiter=',')
        except:
            continue
            
        print(f'Executando para o fold {fold}')
            
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test,  y_test  = dataset_test[:, :-1],  dataset_test[:, -1]

        # Criando as configurações para um dataset específico.
        # Save_file salva em um arquivo as configurações, que são numeradas no
        # report de resultados
        confs = {
            'criterion'         : ['mse'],
            'n_estimators'       : [100, 200, 300],
            'min_samples_split' : [len(X_train)//100, len(X_train)*5//100, len(X_train)*10//100],
        }
        
        RMSE_cv, conf, conf_id = None, None, None
        if len(resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)])>0:
            # Verificar se aquele fold já foi avaliado em alguma repetição, e caso tenha sido
            # pega a configuração utilizada em uma delas (vao ser todas iguais, tanto faz a repetição
            # contanto que seja no mesmo fold)
            aux_resultsDF = resultsDF[(resultsDF['dataset']==ds) &(resultsDF['Fold']==fold)]
            conf_id = aux_resultsDF['conf'].values[0]
            RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]

            keys, values = [], []
            for k,v in confs.items():
                if isinstance(v, list): 
                    values.append(v)
                elif callable(v): 
                    continue
                else:
                    raise Exception('o que é isso?')
                keys.append(k)
                    
            conf    = [dict(zip(keys,items)) for items in product(*values)][conf_id]

            print(f'Pegando configuração avaliada anteriormente: {RMSE_cv}, {conf_id}')
        else:
            # Obtendo melhor configuração para esse treino-teste
            print('Obtendo a melhor configuração utilizando 5-fold cv no treino')

            print(confs)
            # Usando o gridsearch para obtenção da melhor configuração
            
            #cv=5 especifica que os dados devem ser divididos em 2 folds, e realizar uma validação cruzada
            grid = GridSearchCV(
                RandomForestRegressor(),
                confs,
                n_jobs=4,
                cv=5,
                verbose=1,
                scoring=make_scorer(RMSE, greater_is_better=False), # Greater is better vai trocar o sinal e transformar em um problema de maximização. Na prática, isso significa que temos que trocar o sinal na hora de reportar o melhor resultado retornado
                return_train_score=True
            ).fit(X_train, y_train)

            conf    = grid.best_params_
            RMSE_cv = -1*grid.best_score_
            conf_id = np.where(grid.cv_results_['rank_test_score'] ==1)[0][0]

        # Aqui ele não é determinístico. Vamos executar as repetições
        for rep in range(runs_per_fold):
            if len(resultsDF[
                (resultsDF['dataset']==ds) &
                (resultsDF['Fold']==fold)  &
                (resultsDF['Rep']==rep)
            ])==1:
                print(f'already evaluated {ds}-{fold}-{rep}')

                continue
            print(f'evaluating config {conf_id} for {ds}-{fold}-{rep}')

            print(f'evaluating {ds}-{fold}-{rep}')
            
            # Utilizando a melhor configuração para treinar o modelo e obter os scores        
            regressor = RandomForestRegressor(**grid.best_params_).fit(X_train, y_train)  

            RMSE_train = mean_squared_error(regressor.predict(X_train).ravel(), y_train.ravel(), squared=False)
            RMSE_test  = mean_squared_error(regressor.predict(X_test).ravel(), y_test.ravel(), squared=False)

            # Vamos salvar o número da configuração para ficar mais sucinto
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

resultsDF = pd.read_csv(fname)

pd.set_option('display.max_colwidth', None) #não truncar colunas usando display

# Obtendo a melhor configuração para cada dataset

# Tirando a média da melhor configuração em cada fold (e descartando 2 primeiras colunas, configuração e cv)
resultsDF_median = resultsDF.groupby(['dataset']).mean().iloc[:, 4:]
print(resultsDF_median)
resultsDF_median.columns = ['RMSE_train_mean', 'RMSE_test_mean']

# Colocando o desvio padrão e tirando as 2 primeiras colunas (fold e rep, não interessam)
resultsDF_std = resultsDF.groupby(['dataset']).std().iloc[:, 4:]
resultsDF_std.columns = ['RMSE_train_std', 'RMSE_test_std']

# juntando tudo em um só
resultsDF_ = pd.merge(resultsDF_median, resultsDF_std, left_index=True, right_index=True)
print(resultsDF_)