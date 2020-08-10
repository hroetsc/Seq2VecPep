#!/bin/bash

module purge

module load python/3.8.2 gcc/8.2.0 openmpi/gcc/64/3.1.4

cd ~/Cluster/Training/horovod/
source bin/activate
cd ~/Hotspots/

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas/10.1.105
module load cuda10.1/fft/10.1.105
module load cuda10.1/nsight/10.1.105
module load cuda10.1/profiler/10.1.105
module load cudnn/10.1v7.6.5

salloc -p gpu -C scratch2 -N 4 -n 4 --tasks-per-node 1 --gpus-per-task=1 --mem-per-gpu=10G -t 06:00:00 --job-name='hotspots'

scontrol show hostnames $SLURM_JOB_NODELIST

srun --mpi=pmix -o hotspots-%J-%N.out python C_fitModel.py
#mpirun --mca mpi_warn_on_fork 0 --output-filename hotspots-%J-%N.out python C_fitModel.py
