#!/bin/bash

echo "File: $1"

empty=./data/empty_results.csv

cp $empty $1

# number of threads goes from 2 to 1024 in powers of 2
for i in {1..5}
do
	../gpu_match $((2**i)) >> $1
done
