#!/bin/bash

for KEEP in 7 8 9 10:
do
	for WINDOW in 5 6 7 8 9 10 11 12 13 14 15:
	do
		python skip_gram_NN_1.py "$KEEP" "$WINDOW" &
	done
done