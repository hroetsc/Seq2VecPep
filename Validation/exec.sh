#!/bin/bash

for ITER in `seq 50`
do
	echo "THIS IS ITERATION $ITER - STARTING SNAKEMAKE" &
	snakemake --unlock --use-conda -j 64 -R sampling &
done