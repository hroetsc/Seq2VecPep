# Seq2Vec
unsupervised pattern mining for the embedding of biological sequences

## execution ##
`snakemake --use-conda -j -R some_rule --default-res mem_mb=25000`

## file explanation ##
**src/snakefiles** contains all scripts that are in the Snakemake pipeline. `report.html` explains their interdependencies as well as the order of execution in the pipeline.  
All rules are concatenated in the `Snakefile`. Software versions, configurations and input/output files can be accessed via `environment_lab.yaml`, `config.yml` and `features.yml`, respectively.

## upload changes when working remotely ##
clone repository, copy new files in repository  
`git add .`  
`git commit -m "<some comment>"`  
`git push origin master`  

## environment handling ###
Create a virtual environment for Snakemake: `conda env create -f envs/environment_seq2vec.yml`  
Activate the environment: `conda activate seq2vec`  
Execute Snakemake in this environment. Note: The python scripts (`skip_gram_NN_1.py` and `skip_gram_NN_2.py` are executed in different environments which are specified in the rule's conda argument).
