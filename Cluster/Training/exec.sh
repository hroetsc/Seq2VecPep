#!/bin/bash

module purge

module load python/3.8.2
module load gcc/8.2.0
#module load openmpi/gcc/64/4.0.0
module load openmpi/gcc/64/3.1.4

cd horovod/
source bin/activate
cd ..

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas/10.1.105
module load cuda10.1/fft/10.1.105
module load cuda10.1/nsight/10.1.105
module load cuda10.1/profiler/10.1.105
module load cudnn/10.1v7.6.5

salloc -p gpu -C scratch2 -N 20 -n 20 --tasks-per-node 1 --gpus-per-task=1 --mem-per-gpu=30G -t 01-06:00:00 --mail-type=END --mail-user=hanna.roetschke@mpibpc.mpg.de --job-name='seq2vec'

scontrol show hostnames $SLURM_JOB_NODELIST > nodes.txt
scontrol show hostnames $SLURM_JOB_NODELIST

snakemake --unlock --cores 1
snakemake --jobs 100000 --cores 100000 --cluster "srun --mpi=pmix -o hp_training-%J-%N.out" --latency-wait 300 -R training_w5

source deactivate



#HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]
#snakemake --jobs 100000 --cores 100000 --cluster "srun --slurmd-debug=4 --mpi=pmix -o hp_training-%J-%N.out" --latency-wait 300 -R training_w5
