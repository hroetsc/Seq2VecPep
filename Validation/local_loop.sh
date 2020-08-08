for ITER in `seq 6`
do
        snakemake --cores 16 -R data_gen
done
