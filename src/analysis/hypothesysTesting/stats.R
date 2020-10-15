rm(list = ls())

library(xtable)

# final-resultsregression is a table containing the rmses on train and test
# of all algorithms on all datasets. It is obtained by combining the tables inside
# results/rmse. There is a script to create this file inside src/analysis/RMSEs
df <- read.csv("final-resultsregression.csv", header=TRUE, sep=',')

# To change the table column order, just rearange this next 2 lines:
algs     <- c("ITEA", "FEATFull", "SymTree", "GSGP", "GPLearn", "DCGP", "forest", "knn", "tree", "elnet", "lasso", "lassolars", "ridge", "FEAT")
datasets <- c('airfoil', 'concrete', 'energyCooling', 'energyHeating', 'Geographical', 'tecator', 'towerData', 'wineRed', 'wineWhite', 'yacht')

# Reference algorithm for the table
ref_alg <- 'ITEA'

# Matrix to store the comparison in relation to the reference algorithm
final_df <- matrix(nrow = length(algs)-1, ncol = length(datasets))

# Renaming columns so we can extract the results from the p-value table
rownames(final_df) <- algs[algs!=ref_alg]
colnames(final_df) <- datasets

for (dataname in datasets){
   
  dfSel <- df[ which(df$Dataset==dataname & df$Algorithm %in% algs), ]
  
  res <- pairwise.wilcox.test(dfSel$RMSE_test, dfSel$Algorithm, p.adjust.method='bonferroni') 
  
  cap = paste0(dataname)
  
  res$p.value.formatted = format(res$p.value,scientific=T,digits=2)
  
  # Pairwise wilcoxon test with p-value adjustment. With this configuration, we can compare
  # the obtained p-values with the 0.05 value.
  res$p.value.formatted = ifelse(res$p.value < 0.05, paste0("{\\bf ",res$p.value.formatted, "}"),res$p.value.formatted)
  
  ltx <- xtable(res$p.value.formatted, caption=cap,  caption.placement="top",sanitize.text.function = identity)
  
  aux <- c(res$p.value.formatted[ref_alg, ], res$p.value.formatted[ ,ref_alg])
  aux <- aux[!is.na(aux)]
  
  for (alg in algs[algs!=ref_alg]){
    final_df[alg, dataname] = aux[alg]
  }

  #print(ltx,sanitize.text.function = identity, caption.placement="top")
}

ltx <- xtable(final_df, caption="Comparison of test RMSEs of different methods versus the ITEA",  caption.placement="top", sanitize.text.function = identity)
print(ltx,sanitize.text.function = identity, caption.placement="top")