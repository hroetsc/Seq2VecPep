# Seq2Vec
unsupervised pattern mining for the embedding of biological sequences  
[project overview](https://github.com/hroetsc/Seq2Vec/blob/master/Seq2Vec_detailed.svg "Seq2Vec idea")

## Seq2Vec core pipeline
Can be found in the `Seq2Vec` folder. Contains all essential scripts. Scripts that are executed within the Snakemake workflow are in `src` folder.  
Currently, the model training is not part of the pipeline as it is executed separately on the cluster (`Cluster/Training` folder).

## Validation pipeline
Independent Snakemake pipeline to evaluate Seq2Vec embeddings. Needs to be fed with all Seq2Vec embeddings and a biophysical property table.

## Optimization
Fine-tune model hyperparameters using two different approaches
* Gaussian process regression
* Gradient boosting

(Needs to be updated)

## Hotspots ##
Work in progress. Attempts to predict hotspots using Seq2Vec and a convolutional neural network.  
  
*features*: window matrix of *tokPerWindow* x *embeddingDim shape*, protein vector, PCs of window matrix, most informative features selected using sklearn feature selection, ...

### model versions ###
| version | description |
| ------- | ----------- |
| 1       | simple convolutional neural net (CNN) with dense fully-connected layers |
| 2       | [ResNet](http://link.springer.com/10.1007/978-3-319-46493-0_38)-like architecture with optional dilation rates |
| 3       | [DenseNet](https://ieeexplore.ieee.org/document/8099726/) (densely connected CNN) |
| 4       | mixture of ResNet and [SpliceAI](https://linkinghub.elsevier.com/retrieve/pii/S0092867418316295) architecture (skip-connections between residual blocks) |
| 5       | custom model structure: passing each token independently through dense layers, concatenate and apply convolution; merge with convolution on full window matrix |
| 6       | simple ensemble of dense layers --> was used to compare different extensions |
| 7       | using a convolutional autoencoder to learn the inherent structure of embeddings |
| 8       | mixed binary classification and regression |
| 9       | custom structure: applying four different residual block architectures to the input independently --> concatenating and flattening, passing to dense fully-connected layers |
| 10      | custom structure with convolutional autoencoder on PC matrix (more unique than whole window matrix) |
| 11      | simple CNN with 2 input channels: window matrix and protein vector (replicated *tokPerWindow*-times) |
| 12      | capsule neural network ([CapsNet](http://arxiv.org/abs/1710.09829)) for regression |

### additional information ###
* the weights of the best model (lowest validation loss) are restored at the end of the training process and used for prediction
* predictions of all ranks (GPUs) are combined --> pseudo-ensemble
* the learning rate in each epoch follows a cosine annealing schedule  --> find the widest local minimum, can be extended to a real [snapshot ensemble](http://arxiv.org/abs/1803.05407)
* there is an option for adding noise to the input data (brightness change, horizontal and vertical flipping of matrices)

