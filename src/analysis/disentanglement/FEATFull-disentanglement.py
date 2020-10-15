import pandas as pd
import numpy  as np

import myParser
import re

from scipy import stats


# Pre processing the string to work with the parser
# for the FeatFull results file
def Z_FEATFull(feat_str, Xtrain):
    
    nsamples, nvars = Xtrain.shape

    # Changing the sqrt(|x|) 
    feat_str = feat_str.replace('sqrt(|', 'SQRTABS((')
    feat_str = feat_str.replace('|', ')')
    
    feat_str = feat_str.replace("if-then-else", "if")
    
    # replace >= and <= with geq and leq, avoiding matching with > or <
    feat_str = feat_str.replace('>=', ' geq ')
    feat_str = feat_str.replace('<=', ' leq ')
    feat_str = feat_str.replace('==', ' eq ')

    feat_strs = re.findall(r'\[([^]]*)\]', feat_str)

    nterms = len(feat_strs)
        
    Z = np.zeros( (nsamples, nterms) )
    
    for col, feat in enumerate(feat_strs):
         Z[:, col] = myParser.parse(feat, Xtrain)

    disentanglements = []
    
    for col, _ in enumerate(feat_strs):
        for comparison, _ in enumerate(feat_strs):
            if col != comparison:
                corr, p = stats.pearsonr( Z[:, col], Z[:, comparison] )
                
                # Pearson's correlation divides by the std, and the
                # existante of a result is not guaranted. Handling possible NaNs
                corr = 0.0 if np.isnan(corr) else corr
                
                disentanglements.append(corr**2)
        
    return np.mean(disentanglements)


if __name__ == '__main__':
    datasets = ["airfoil", "concrete", "energyCooling", "energyHeating", "yacht", "Geographical", "towerData", "tecator", "wineRed", "wineWhite"]

    print("Dataset,Fold,RMSE_train,RMSE_test,expr,disentanglement")

    df = pd.read_csv('../../../results/rmse/FEATFull-resultsregression.csv')
    vs = df[["Dataset","Fold","RMSE_train","RMSE_test", "Expression"]].values
    m, n = vs.shape

    for dname, fold, RMSE_train, RMSE_test, Expression in vs:
        df_train = pd.read_csv(f'../../../datasets/commaSeparated/{dname}-train-{fold}.dat', sep=',', header=None)

        Xtrain = df_train.iloc[:, :-1].values

        disentanglement = 'nan'
        try:
            disentanglement = Z_FEATFull(Expression, Xtrain)
        except Exception as e:
            print(e)
            continue

        print(f"{dname}, {fold}, {RMSE_train}, {RMSE_test}, '{Expression}', {disentanglement}")
