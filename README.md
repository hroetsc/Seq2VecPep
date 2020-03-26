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

## environment handling ##
To create a new environment from a `.yml` file in the `envs/` folder, enter:  
`conda env create -f environment_seq2vec.yml`  
To activate the environment, enter: `conda activate seq2vec`.  
The Python scripts run in different environments specified by `.yml` files in the `src/snakefiles/` folder.
