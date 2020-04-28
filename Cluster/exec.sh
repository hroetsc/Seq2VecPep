#!/bin/bash

scdir=/scratch2/hroetsc/Seq2Vec/
hdir=/usr/users/hroetsc/Cluster

module purge
module load conda/4.3.30

source activate seq2vec

cp -rf $hdir/data $scdir
cp -rf $hdir/results $scdir

#snakemake --use-conda -j 500 --allowed-rules skipgrams --cluster "sbatch -A all -o hp_w3_d100.out -p fat -C scratch2 -G 0 -N 15 -n 500 -t 01:00:00 --mail-type=END --mail-user=hanna.roetschke@mpibpc.mpg.de"

snakemake --use-conda -j 300 --cluster "sbatch -A all -o hp_w3_d100-%J.out -p gpu -C scratch2 -G 30 -N 16 -n 60 -t 01-00:00:00 --mail-type=END --mail-user=hanna.roetschke@mpibpc.mpg.de"

cp -rf $scdir/Cluster ~
rm -rf $scdir/Cluster

source deactivate
