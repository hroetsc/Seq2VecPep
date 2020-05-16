for ITER in `seq 15`
do
        cp -rf src/ $ITER &&
        cd $ITER &&
        snakemake --unlock &&
        snakemake --use-conda -j 64 -R evaluation &
done
