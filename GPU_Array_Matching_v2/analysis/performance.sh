#!/bin/bash

echo "File: $1"

empty=./data/empty_results.csv

cp $empty $1

# number of threads goes from 2 to 1024 in powers of 2 with 3 runs at each number of threads
for i in {1..5}
do
	../gpu_match $((2**i)) >> $1
	../gpu_match $((2**i)) >> $1
	../gpu_match $((2**i)) >> $1
done
