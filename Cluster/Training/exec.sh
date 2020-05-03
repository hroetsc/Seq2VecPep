#!/bin/bash

module purge

module load conda/4.3.30

source activate seq2vec

module load cuda10.0/toolkit/10.0.130
module load cudnn/10.0v7.6.3


snakemake --unlock
snakemake --use-conda -j 64 --cluster "srun -A all -o hp_training-%J.out -p gpu -C scratch2 -N 10 -n 200 -G 20 -t 2-00:00:00 --mail-type=END --mail-user=hanna.roetschke@mpibpc.mpg.de"

source deactivate
