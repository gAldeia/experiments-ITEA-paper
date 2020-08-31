import os
import glob

import os.path as path
import pandas  as pd
import numpy   as np

from itertools         import product
from sklearn.metrics   import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from IPython.display         import display, Markdown, Latex

cur_folder = os.getcwd() # Diretório atual

datasets_folder = '../datasets/'


# Função de erro para o gridsearch 
def RMSE(yhat, y):
    return np.sqrt(np.square(yhat - y).mean())
    
n_folds         = 5
max_n_neighbors = 30 # Porcentagem máxima de vizinhos (0-100)

# kNN é determinístico, não precisamos fazer repetições
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
columns = ['dataset','conf','Fold','RMSE_train','RMSE_test']

fname = '../docs/knn-resultsregression.csv'
results = {c:[] for c in columns}

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
            
        # Retomar testes se interrompido
        if os.path.isfile(fname):
            resultsDF = pd.read_csv(fname)
            results   = resultsDF.to_dict('list')

            if len(resultsDF[
                (resultsDF['dataset']==ds) &
                (resultsDF['Fold']==fold)
            ])==1:
                print(f'already evaluated {ds}-{fold}')

                continue
            
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test,  y_test  = dataset_test[:, :-1],  dataset_test[:, -1]
        
        # Criando as configurações para um dataset específico.
        # Save_file salva em um arquivo as configurações, que são numeradas no
        # report de resultados
        confs = {
            'n_jobs'      : [None], #não rodar gridsearch e o fit do regressor em paralelo para não dar problema
            'weights'     : ['distance', 'uniform'],
            'n_neighbors' : list(range(1, (len(X_train)*max_n_neighbors)//100))
        }

        print(confs)
        # Usando o gridsearch para obtenção da melhor configuração
        
        print(f'evaluating {ds}-{fold}')
        
        #cv=5 especifica que os dados devem ser divididos em 2 folds, e realizar uma validação cruzada
        grid = GridSearchCV(
            KNeighborsRegressor(),
            confs,
            n_jobs=4,
            cv=5,
            verbose=1,
            scoring=make_scorer(RMSE, greater_is_better=False), # Greater is better vai trocar o sinal e transformar em um problema de maximização. Na prática, isso significa que temos que trocar o sinal na hora de reportar o melhor resultado retornado
            return_train_score=True
        ).fit(X_train, y_train)

        # Utilizando a melhor configuração para treinar o modelo e obter os scores        
        regressor = KNeighborsRegressor(**grid.best_params_).fit(X_train, y_train)  

        RMSE_train = -1*grid.best_score_ #Melhor score no gridsearch corresponde ao treino
        RMSE_test  = mean_squared_error(regressor.predict(X_test).ravel(), y_test.ravel(), squared=False)

        # Vamos salvar o número da configuração para ficar mais sucinto
        results['dataset'].append(ds)
        results['conf'].append(grid.best_params_)
        results['RMSE_train'].append(RMSE_train)
        results['RMSE_test'].append(RMSE_test)
        results['Fold'].append(fold)

        df = pd.DataFrame(results)
        df.to_csv(fname, index=False)

print('done')

fname = '../docs/knn-resultsregression.csv'

resultsDF = pd.read_csv(fname)

pd.set_option('display.max_colwidth', None) #não truncar colunas usando display

# Obtendo a melhor configuração para cada dataset

# Calculando as medianas e tirando a coluna de fold (configuração já some pois não é numérico)
resultsDF_median = resultsDF.groupby('dataset').mean().iloc[:, 1:]
resultsDF_median.columns = ['RMSE_train_mean', 'RMSE_test_mean']

# Colocando o desvio padrão 
resultsDF_std = resultsDF.groupby('dataset').std().iloc[:, 1:]
resultsDF_std.columns = ['RMSE_train_std', 'RMSE_test_std']

# juntando tudo em um só
resultsDF_ = pd.merge(resultsDF_median, resultsDF_std, left_index=True, right_index=True)
print(resultsDF_)
