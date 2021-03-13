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

# array_size goes from 8 to 8192 in increments of 8
for i in {8..8192..8}
do	
	./main $(i) 8 1 >> $file1
	./main $(i) 8 0 >> $file1

	./main $(i) 512 1 >> $file2
	./main $(i) 512 0 >> $file2

	./main $(i) 1024 1 >> $file3
	./main $(i) 1024 0 >> $file3
done

# num_arrays goes from 8 to 1024 in increments of 8
for j in {8..1024..8}
do		
	./main 8 $(j) 1 >> $file4
	./main 8 $(j) 0 >> $file4
	
	./main 512 $(j) 1 >> $file5
	./main 512 $(j) 0 >> $file5

	./main 8192 $(j) 1 >> $file6
	./main 8192 $(j) 0 >> $file6
done
