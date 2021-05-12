#!/bin/bash

file1=./data/change_arr_size_small.csv
file2=./data/change_arr_size_avg.csv
file3=./data/change_arr_size_lg.csv
file4=./data/change_num_arr_small.csv
file5=./data/change_num_arr_avg.csv
file6=./data/change_num_arr_lg.csv

empty=./data/empty_results.csv

cp $empty $file1
cp $empty $file2
cp $empty $file3
cp $empty $file4
cp $empty $file5
cp $empty $file6

#Current maximum size is 1024 * 1024

# array_size goes from 4 to 1024 in increments of 4
for i in {4..1024..4}
do
	../main $((i)) 4 1 0 >> $file1
	../main $((i)) 4 0 0 >> $file1

	../main $((i)) 512 1 0 >> $file2
	../main $((i)) 512 0 0 >> $file2

	../main $((i)) 1024 1 0 >> $file3
	../main $((i)) 1024 0 0 >> $file3
done

# num_arrays goes from 4 to 1024 in increments of 4
for i in {4..1024..4}
do
	../main 4 $((i)) 1 0 >> $file4
	../main 4 $((i)) 0 0 >> $file4

	../main 512 $((i)) 1 0 >> $file5
	../main 512 $((i)) 0 0 >> $file5

	../main 1024 $((i)) 1 0 >> $file6
	../main 1024 $((i)) 0 0 >> $file6
done
