# Noise-Resistant-Encoding

## Paper and Data
This repository contains code for our paper "Noise Resistant Deep Entity Matching".
All public datasets used in
the paper can be downloaded from the [datasets page](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).

## NRE as preprocessing
Our proposed Noise Resistant Encoding (NRE) can be seen as a preprocessing stage before inputting to the downstream classifier *ditto*. 
Running NRE consist of following steps.

1. Running NRE Clustering over a EM training set and encode the training set.
    ```
        python3 nre/nre_clustering.py 
    ```
    You can set hyperparameters by changing global variables in nre/nre_clustering.py.

2. Encoding the validation set and test set.
## Nov. 1, 2020: a few days are required to update this repo.

