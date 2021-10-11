#!/bin/bash

file1=./data/change_t_a4.csv

empty=./data/empty_results.csv

cp $empty $file1

# array_size goes from 4 to 1024 in increments of 4
for i in {4..1024..8}
do
	../gpu_match $((i)) 0 >> $file1
	../gpu_match $((i)) 0 >> $file1
	../gpu_match $((i)) 0 >> $file1
done
