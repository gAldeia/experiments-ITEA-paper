## pasta datasets:

contém os dados já separados em 5 folds de treino e teste, salvos com extensão .dat, com valores separados por vírgula. Estão no formato para uso 

## Pasta docs/datasets_without_traintestsplit:

Quando baixar um novo dataset [desse site](https://github.com/EpistasisLab/penn-ml-benchmarks), colocar o arquivo .tsv ali. É uma pasta onde guardamos datasets para preparar para uso.

## src/notebooks:

* Criação de folds: Pega os datasets da pasta docs/datasets_without_traintestsplit, aplica uma divisão em 5 folds e salva os arquivos gerados no formato correto dentro de datasets
* kNN gridsearch: faz o gridsearch do knn, e salva os resultados em docs/knn-results.csv

## src/GSGP/

pasta com a modificação do gsgp para ter funções de transformação iguais às utilizadas no ITEA.