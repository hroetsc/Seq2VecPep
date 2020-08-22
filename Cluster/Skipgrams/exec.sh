#!/bin/bash

scdir=/scratch2/hroetsc/Seq2Vec
hdir=/usr/users/hroetsc/Cluster

module purge

module load conda/4.3.30

source activate seq2vec

cp -rf $hdir/data $scdir

snakemake --unlock
snakemake --use-conda -j 200 --cluster "sbatch -A all -o hp_skipgrams-%J.out -p gpu -C scratch2 -N 10 -n 200 -G 30 -t 2-00:00:00 --mail-type=END --mail-user=hanna.roetschke@mpibpc.mpg.de"

cp -rf $scdir/data $hdir
cp -rf $scdir/results $hdir

source deactivate
