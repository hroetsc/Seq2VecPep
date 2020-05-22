for ITER in `seq 15`
do

	mkdir $ITER &&
	cp -rf data/ $ITER &&
	cp -rf src/ $ITER &&
	cp -rf Snakefile $ITER &&
	cp -rf features.yaml $ITER &&

	cd $ITER &&
	pwd &&

	echo "THIS IS ITERATION $ITER - STARTING SNAKEMAKE" &&

	snakemake --unlock &&
	snakemake --use-conda -j 24 -R sampling &

done
