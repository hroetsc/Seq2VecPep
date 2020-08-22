# Seq2Vec
unsupervised pattern mining for the embedding of biological sequences

![](Seq2Ved_detailed.ps)

## Seq2Vec core pipeline
Can be found in the `Seq2Vec` folder. Contains all essential scripts. Scripts that are executed within the Snakemake workflow are in `src` folder.  
Currently, the model training is not part of the pipeline as it is executed separately on the cluster (`Cluster` folder).

## Validation pipeline
Independent Snakemake pipeline to evaluate Seq2Vec embeddings. Needs to be fed with all Seq2Vec embeddings and a biophysical property table.

## Optimization
Fine-tune model hyperparameters using two different approaches
* Gaussian process regression
* Gradient boosting

(Needs to be updated)

## Hotspots ##
Work in progress. Attempts to predict hotspots using Seq2Vec.
