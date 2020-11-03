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
    ```
    python3 nre/nre_encoding.py
    ```
    You can apply Noise Model on a test set by running utilities/add_noise.py

## Training and Testing on *Ditto*
The encoded training, validation, and test set are used for *ditto*.

1. For a dataset contains tuple pairs consist of more than 512 tokens (Textual Company), run utilities/tfidf_summarization.py to summarize it.
    (Noise Model is applied after this summarization.)
    
2. Running utilities/mixDA.py to generate a augmented training set for *ditto*'s mixDA.

3. Running utilities/span_typing_parallel.py for injecting domain knowledge to training, validation, and test set.

## Nov. 1, 2020: a few days are required to update this repo.

