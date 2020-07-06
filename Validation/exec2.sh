for ITER in `seq 15`
do
        cd $ITER &&
        pwd &&
        snakemake --unlock --cores 64 &&
        snakemake --use-conda --cores 64 -R data_gen &
done
