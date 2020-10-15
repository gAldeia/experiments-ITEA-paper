import numpy as np
import pandas as pd

files = [
    # Genetic algorithms
    'ITEA-resultsregression',
    'FEATFull-resultsregression',
    'FEAT-resultsregression',
    'SymTree-resultsregression',
    
    'GSGP-resultsregression',
    'GPLearn-resultsregression',
    'DCGP-resultsregression',
    
    # Non parametric regressions
    'forest-resultsregression',
    'knn-resultsregression',
    'tree-resultsregression',
    
    # linear regressions
    'elnet-resultsregression',
    'lasso-resultsregression',
    'lassolars-resultsregression',
    'ridge-resultsregression',
]

rmses_path = '../../../results/rmse/'

dfs = []
for file in files:
    df = pd.read_csv(f'{rmses_path + file}.csv')
    
    df['Algorithm'] = file.replace('-resultsregression', '')
    
    try:
        df = df.rename(columns={'dataset': 'Dataset'})
    except:
        pass
    
    df = df[['Algorithm', 'Dataset', 'RMSE_train', 'RMSE_test']]
    dfs.append(df)
    
final_df = pd.concat(dfs, axis=0)
final_df.to_csv(f'{rmses_path}final-resultsregression.csv', index=None)