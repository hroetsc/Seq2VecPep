for VAR in 1000 5000 10000 50000 100000
do
	python skip_gram_NN_2.py "$VAR" &
done