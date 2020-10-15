import pandas as pd
import numpy  as np

import myParser
import re

from scipy import stats


# Pre processing the string to work with the parser
# for the symTree results file
def Z_IT(itexpr_str, Xtrain):
    
    nsamples, nvars = Xtrain.shape
    
    # Avoidind problems with the dot on the parser
    itexpr_str = itexpr_str.replace('sqrt.abs', 'SQRTABS')
    
    # variables should be labeled x_{i}
    for i in range(nvars):
        itexpr_str = itexpr_str.replace(f'x_{i+1}', f'x_{i}')

    # Removing coefficients and intercept
    itexpr_str = re.sub(r'^-?[\d.]+(e-?\d+)?\s\+\s', '', itexpr_str)
    itexpr_str = re.sub(r'-?[\d.]+(e-?\d+)?\*', '', itexpr_str)

    # obtaining the terms
    itexpr_its = itexpr_str.split(' + ')

    nterms = len(itexpr_its)

    Z = np.zeros( (nsamples, nterms) )
    for col, it in enumerate(itexpr_its):
        Z[:, col] = myParser.parse(it, Xtrain)

    # Store all disentanglements calculated
    disentanglements = []
    
    for col, _ in enumerate(itexpr_its):
        for comparison, _ in enumerate(itexpr_its):
            if col != comparison:
                corr, p = stats.pearsonr( Z[:, col], Z[:, comparison] )
                
                # Pearson's correlation divides by the std, and the
                # existante of a result is not guaranted. Handling possible NaNs
                corr = 0.0 if np.isnan(corr) else corr
                
                disentanglements.append(corr**2)
        
    return np.mean(disentanglements)




if __name__ == '__main__':
    datasets = ["airfoil", "concrete", "energyCooling", "energyHeating", "yacht", "Geographical", "towerData", "tecator", "wineRed", "wineWhite"]

    # Reading the results without disentanglement and printing them with the disentanglement
    df   = pd.read_csv(f"../../../results/rmse/SymTree-resultsregression.csv")
    vs   = df[["dataset","fold","RMSE_train","RMSE_test", "expr"]].values
    m, n = vs.shape

    print("Dataset, Fold, RMSE_train, RMSE_test, expr, disentanglement")
    for dname, fold, RMSE_train, RMSE_test, Expression in vs:
        df_train = pd.read_csv(f'../../../datasets/commaSeparated/{dname}-train-{fold}.dat', sep=',', header=None)

        Xtrain = df_train.iloc[:, :-1].values
        
        disentanglement = 'nan'

        try:
            disentanglement = Z_IT(Expression, Xtrain)
        except Exception as e:
            continue

        print(f"{dname},{fold},{RMSE_train},{RMSE_test},{Expression},{disentanglement}")