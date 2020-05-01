#!/bin/bash

module purge

module load cuda10.0/toolkit/10.0.130
module load conda/4.3.30

source activate seq2vec

snakemake --unlock
snakemake --use-conda -j 200 --cluster "sbatch -A all -o hp_training-%J.out -p gpu -C scratch2 -N 10 -n 200 -G 30 -t 2-00:00:00 --mail-type=END --mail-user=hanna.roetschke@mpibpc.mpg.de"

source deactivate
