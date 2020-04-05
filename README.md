# Seq2Vec
unsupervised pattern mining for the embedding of biological sequences

## execution ##
Make sure you installed Miniconda.  
Clone this repository. Within the Seq2Vec folder, create a virtual Snakemake environment:  
`conda env create -f envs/environment_seq2vec.yml`  
  
Activate the environment:  
`conda activate seq2vec`  
If necessary, install or update Snakemake:  
`conda install -c conda-forge -c bioconda snakemake`  
Adjust hyperparameters in the `hyperparams.csv` file (this is not how it should be, will be improved in the future).  
  
Execute the pipeline:  
`snakemake --use-conda -j <number of threads> -R <some rule, in case you want to re-run anything>`  
If you are done, type: `conda deactivate` to return to your base environment.


## hyperparameters ##
- `threads`: self-explanatory. The number of threads used by the pipeline. Note that the Snakemake `-j` flag is dominant over this hyperparameter.  
- `Seqinput`: .csv file which must contain at least two columns: `Accession` and `seqs` (see file explanation)  
- `BPEinput`: .fasta file which is used to train the BPE algorithm. The default is the whole reviewed UniProtKB (SwissProt). 
- `BPEvocab`: vocab size for the byte-pair encoding algorithm.
- `keep`: skip-grams are downsampled to reduce the complexity of the model / the training time. Specify the fraction of skip-grams you wish to keep per target word integer per protein. This strongly depends on the size of your dataset.  
- `negSkipgrams`: relation of negative word pairs to positive pairs, e.g. 1 means 50 % negative and 50 % positive samples.  
- `windowSize`: size of the frame in which a token is considered as context word. The context of a target word is: `[target - windowSize ; target + windowSize +1]`  
- `embedding`: dimensionality of the vector representation  
- `epochs`: number of training epochs  
- `valSplit`: fraction of skip-grams used for model validation  
- `batchSize`: As the whole skip-gram corpus would not fit into memory, the model is trained on batches.

## file explanation ##
**src/** contains all scripts that are in the Snakemake pipeline. `report.html` explains their interdependencies as well as the order of execution in the pipeline.  
All rules are concatenated in the `Snakefile`. Software versions and input/output files can be accessed via `environment_lab.yaml` and `features.yml`, respectively. 
Hyperparameters are currently stored in `hyperparams.csv`
  
**input requirements** The input file (`Seqinput` in `hyperparams.csv` must contain at least two columns: `Accession` and `seqs` which contain the name of a protein/transcript and its sequence, respectively.

## environment handling ###
Note: The python scripts (`skip_gram_NN_1.py` and `skip_gram_NN_2.py` are executed in different environments which are specified in the rule's conda argument).
