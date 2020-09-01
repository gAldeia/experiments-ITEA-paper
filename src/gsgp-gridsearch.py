import os.path   as path
import os
import glob
import pandas as pd
import numpy as np
from itertools import product

# Diretório atual
cur_folder = os.getcwd()

adapted_datasets_folder  = '../datasets_GSGP'

# compilar o código fonte
os.system(f'cd "{cur_folder}/"')
os.system(f'g++ GP.cc -o GP.exe')


# Função para atualizar o arquivo de configuração
# de acordo com um dicionario de configuração recebido
def update_config(conf, pred=False):
    # Pred determina se é para gerar a configuração com ou sem
    # o uso dos resultados da evolução
    
    data = []
    with open(f'{cur_folder}/configuration.ini', 'r+') as f:
        data = f.read().splitlines()

    for line_id, line in enumerate(data):
        param = line.split(' ')[0]
        value = line.split(' ')[-1]
        
        if param == 'expression_file': #caso seja para executar teste
            data[line_id] = line.replace(str(value), f'{1 if pred else 0}\n')
        elif param == 'USE_TEST_SET': # caso seja para executar teste
            data[line_id] = line.replace(str(value), f'{1 if pred else 0}\n')
        else: # copio os outros valores
            data[line_id] = line.replace(str(value), f'{conf[param]}\n')

    with open(f'{cur_folder}/configuration.ini', 'w') as f:
        f.writelines( data )
        

# Função que recebe partições de treino e validação (5-fold), e uma
# lista de configurações, e determina a melhor configuração. 
# Retorna o RMSE médio da melhor configuração, um dicionário da melhor
# configuração, e o id da melhor configuração.
# A configuração encontrada será utilizada em todas as repetições de uma divisão treino-teste
def gridsearch(dataset_train_cv, dataset_validation_cv, confs):
     
    # (rmse_cv, configuração, indice da configuração)
    best_conf = (np.inf, None, -1)
    
    for i, conf in enumerate(confs):
        update_config(conf, pred=False)
        
        print(f'Testando configuração {i}/{len(confs)}')
        
        RMSE_cv = []
        for train_cv, validation_cv in zip(dataset_train_cv, dataset_validation_cv):
            os.system(f'./GP.exe -train_file {train_cv} -test_file {validation_cv}')
            
            cv_test = pd.read_csv(f'{cur_folder}/fitnesstest.txt', header=None)
            RMSE_cv.append(cv_test.iloc[-1, 0])

        if np.mean(RMSE_cv) < best_conf[0]:
            best_conf = (np.mean(RMSE_cv), conf,  i)
            
    return best_conf

        
# Função que executa uma configuração específica para uma divisão
# de treino e teste.
def run(dataset_train, dataset_test, conf=None):
    # Executando para os dados fornecidos
    
    # Modificando o arquivo de configuração para ficar igual à configuração
    # sendo utilizada
    update_config(conf, pred=False)
    
    os.system(f'./GP.exe -train_file {dataset_train} -test_file {dataset_test}')

    # Pegando o melhor fitness no treino e no teste (
    # esses arquivos guardam o melhor em cada geração, então a última linha
    # é o melhor)
    
    fitness_train    = pd.read_csv(f'{cur_folder}/fitnesstrain.txt', header=None)
    fitness_test     = pd.read_csv(f'{cur_folder}/fitnesstest.txt', header=None)
    
    return (
        fitness_train.iloc[-1, 0],
        fitness_test.iloc[-1, 0],
    )

# Criando diferentes configurações:

# Listas com valores diferentes para testar no gridsearch.
# são permitidos: valores únicos, listas (será feito um produto cartesiano
# entre as listas), e funções lambda (que serão calculadas para cada
# configuração do produto cartesiano)
gridsearch_configurations = {
    'population_size'        : [100, 250, 500],
    'max_number_generations' : lambda conf:  100000//conf['population_size'],
    'init_type'              : 2,
    'p_crossover'            : [0.2, 0.5, 0.8],
    'p_mutation'             : lambda conf: 1 - conf['p_crossover'],
    'max_depth_creation'     : 5,
    'tournament_size'        : 4,
    'zero_depth'             : 0,
    'mutation_step'          : 1.0, 
    'num_random_constants'   : 0,
    'min_random_constant'    : -100,
    'max_random_constant'    : 100,
    'minimization_problem'   : 1,
    'random_tree'            : 500
    
    # As de baixo ficam reservadas para o script controlar. Elas são relacionadas a
    # executar a evolução ou usar um resultado encontrado
    
    #'expression_file'        : 0,
    #'USE_TEST_SET'           : 0
}

varying = []

keys, values = [], []
for k,v in gridsearch_configurations.items():
    if isinstance(v, list): # parâmetro é uma lista de valores possíveis
        values.append(v)
        varying.append(k)
    elif (isinstance(v, int) or isinstance(v, float)): # parâmetro é um valor único (fixado). Vamos colocar em uma lista para usar o product
        values.append([v])
    elif callable(v): # É uma função, fica pra depois
        continue
    else:
        raise Exception('o que é isso?')
    keys.append(k)
        
# Gerando todas as configurações que serão testadas
confs = [dict(zip(keys,items)) for items in product(*values)]

# Agora vamos pegar as funções lambda e aplicar
for conf in confs:
    for k,v in gridsearch_configurations.items():
        if callable(v): # É uma função, fica pra depois
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
confs_df.to_csv('GSGP-new-gridsearch_configurations.csv')

# Destacando apenas os parâmetros que são diferentes entre algumas configurações
confs_df.style.apply(
    lambda x: ['background: lightgreen' if x.name in varying else '' for i in x], 
    axis=1
)

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

# ---------------------------
columns = ['dataset','conf','Fold','Rep','RMSE_cv','RMSE_train','RMSE_test']

fname = '../docs/GSGP-resultsregression.csv'
results = {c:[] for c in columns}

if os.path.isfile(fname):
    resultsDF = pd.read_csv(fname)
    results   = resultsDF.to_dict('list')


for ds in datasets:
    print(f'Executando agora para o dataset {ds}')
    for fold in range(n_folds):
        dataset_train_cv      = [f'{adapted_datasets_folder}/{ds}-train-{fold}-train-{i}.txt' for i in range(5)]
        dataset_validation_cv = [f'{adapted_datasets_folder}/{ds}-train-{fold}-validation-{i}.txt' for i in range(5)]
        
        dataset_train         = f'{adapted_datasets_folder}/{ds}-train-{fold}.txt'
        dataset_test          = f'{adapted_datasets_folder}/{ds}-test-{fold}.txt'

        # Verificar se os datasets existem
        if len(glob.glob(dataset_train))==0:
            print(f'Dataset {dataset_train} does not exist.')
            continue
        print(f'Executando para o fold {fold}')

        # Verificar se aquele fold já foi avaliado em alguma repetição, e caso tenha sido
        # pega a configuração utilizada em uma delas (vao ser todas iguais)
        RMSE_cv, conf, conf_id = None, None, None
        if os.path.isfile(fname):
            resultsDF = pd.read_csv(fname)
            results   = resultsDF.to_dict('list')

            if len(resultsDF[
                    (resultsDF['dataset']==ds) &
                    (resultsDF['Fold']==fold)
                ])>0:
                aux_resultsDF = resultsDF[
                    (resultsDF['dataset']==ds) &
                    (resultsDF['Fold']==fold)
                ]
                conf_id = aux_resultsDF['conf'].values[0]
                RMSE_cv = aux_resultsDF['RMSE_cv'].values[0]
                conf = confs[conf_id]

                print(f'Pegando configuração avaliada anteriormente: {RMSE_cv}, {conf_id}')

        # Obtendo melhor configuração para esse treino-teste
        if RMSE_cv == conf == conf_id == None:
            print('Obtendo a melhor configuração utilizando 5-fold cv no treino')
            RMSE_cv, conf, conf_id = gridsearch(dataset_train_cv, dataset_validation_cv, confs)

        print('Começando a testar a melhor configuração obtida')
        for rep in range(runs_per_fold):
            if os.path.isfile(fname):
                resultsDF = pd.read_csv(fname)
                results   = resultsDF.to_dict('list')

                if len(resultsDF[
                    (resultsDF['dataset']==ds) &
                    (resultsDF['Fold']==fold)  &
                    (resultsDF['Rep']==rep)
                ])==1:
                    print(f'already evaluated {ds}-{fold}-{rep}')

                    # Importante notar que ele vê a configuração apenas
                    # pelo ID dela. Se modificar as possíveis configurações, vai gerar bugs.
                    # A ideia aqui é apenas retomar um experimento parado no meio, mas que
                    # não tenha sido modificado durante a pausa.
                    continue

            print(f'evaluating config {conf_id} for {ds}-{fold}-{rep}')
            
            RMSE_train, RMSE_test = run(dataset_train, dataset_test, conf)


            results['dataset'].append(ds)

            # Vamos salvar o número da configuração para ficar mais sucinto
            results['conf'].append(conf_id)
            results['RMSE_cv'].append(RMSE_cv)
            results['RMSE_train'].append(RMSE_train)
            results['RMSE_test'].append(RMSE_test)
            results['Fold'].append(fold)
            results['Rep'].append(rep)

            df = pd.DataFrame(results)
            df.to_csv(fname, index=False)

print('done')


# Limpando arquivos que o GP.exe cria
files = ['fitnesstest.txt', 'fitnesstrain.txt', 'trace.txt']
for f in files:
    try:
        os.remove(f'{cur_folder}/{f}')
    except:
        continue
    
fname = '../docs/GSGP-resultsregression.csv'

resultsDF = pd.read_csv(fname)

pd.set_option('display.max_colwidth', None) #não truncar colunas usando display


# Obtendo a melhor configuração para cada dataset


# Pegar, para cada dataset-fold-rep, a configuração de menor RMSE_cv
resultsDF_ = resultsDF.loc[resultsDF.groupby(['dataset', 'Fold', 'Rep'])['RMSE_cv'].idxmin()]
resultsDF_ = resultsDF_.set_index(['dataset', 'Fold', 'Rep'])

# Tirando a média da melhor configuração em cada fold (e descartando 2 primeiras colunas, configuração e cv)
resultsDF_median = resultsDF_.groupby(['dataset']).mean().iloc[:, 1:]
resultsDF_median.columns = ['RMSE_cv', 'RMSE_train_mean', 'RMSE_test_mean']

# Colocando o desvio padrão e tirando as 2 primeiras colunas (fold e rep, não interessam)
resultsDF_std = resultsDF_.groupby(['dataset']).std().iloc[:, 1:]
resultsDF_std.columns = ['RMSE_cv', 'RMSE_train_std', 'RMSE_test_std']

# juntando tudo em um só
resultsDF_ = pd.merge(resultsDF_median, resultsDF_std, left_index=True, right_index=True)
print(resultsDF_)