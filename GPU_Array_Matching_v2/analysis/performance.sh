#!/bin/bash

echo "File: $1"

empty=./data/empty_results.csv

cp $empty $file1

# number of threads goes from 2 to 1024 in powers of 2
for i in {1..10}
do
	../gpu_match $((2**i)) 0 >> $1
	../gpu_match $((2**i)) 0 >> $1
	../gpu_match $((2**i)) 0 >> $1
done
