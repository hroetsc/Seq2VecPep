for ITER in `seq 15`
do

	mkdir $ITER &&
	cp -rf data/ $ITER &&
	cp -rf src/ $ITER &&
	cp -rf Snakefile $ITER &&
	cp -rf features.yaml $ITER &&
	cp -rf .snakemake/ $ITER &&

	cd $ITER &&
	pwd &&

	echo "THIS IS ITERATION $ITER - STARTING SNAKEMAKE" &&

	snakemake --unlock --cores 64 &&
	snakemake --use-conda --cores 64 &

done
