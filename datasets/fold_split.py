# Website suggestion to get new data sets:
# https://epistasislab.github.io/penn-ml-benchmarks/

import numpy as np

import os
import glob

from sklearn.model_selection import KFold


# Will get all datasets in the specified folder
input_folder = '.'
output_folder = '.'
for dataset in glob.glob(f'{input_folder}/*'):
    
    # Extracting the name of the dataset (removing path and file extension)
    file = dataset.replace(input_folder, '').replace('.tsv', '')

    print(file)
    
    # Pay attention to the delimiter. This is defined to work with
    # data sets downloaded from the suggested website
    df = np.loadtxt(dataset, delimiter='\t', skiprows=1)
    X, y = df[:, :-1], df[:, -1]
    
    # Splitting and shuffling
    kf = KFold(n_splits=5, shuffle=True)
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        # saving as a comma separated file
        np.savetxt(f'../datasets/{file}-train-{i}.dat', X[train_index], delimiter=',')
        np.savetxt(f'../datasets/{file}-test-{i}.dat', X[test_index], delimiter=',')