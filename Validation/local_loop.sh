for ITER in `seq 20`
do
        snakemake --use-conda --cores 16 -R data_gen
done
