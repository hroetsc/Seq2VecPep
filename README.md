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
Adjust hyperparameters in the hyperparams.csv file (this is not how it should be, will be improved in the future). I will also add an explanation for all hyperparameters.  
Execute the pipeline:  
`snakemake --use-conda -j <number of threads> -R <some rule, in case you want to re-run anything>` or  
`snakemake --use-conda --cluster qsub -j <number of jobs>`  
If you are done, type: `conda deactivate` to return to your base environment.

## file explanation ##
**src/** contains all scripts that are in the Snakemake pipeline. `report.html` explains their interdependencies as well as the order of execution in the pipeline.  
All rules are concatenated in the `Snakefile`. Software versions and input/output files can be accessed via `environment_lab.yaml` and `features.yml`, respectively. 
Hyperparameters are currently stored in `hyperparams.csv`

## environment handling ###
Note: The python scripts (`skip_gram_NN_1.py` and `skip_gram_NN_2.py` are executed in different environments which are specified in the rule's conda argument).
