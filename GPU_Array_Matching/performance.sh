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

# array_size goes from 8 to 8096
for i in {3..13}
do	
	./main $((2**i)) 8 1 >> $file1
	./main $((2**i)) 8 0 >> $file1

	./main $((2**i)) 512 1 >> $file2
	./main $((2**i)) 512 0 >> $file2

	./main $((2**i)) 1024 1 >> $file3
	./main $((2**i)) 1024 0 >> $file3
done

# num_arrays goes from 8 to 1024
for j in {3..10}
do		
	./main 8 $((2**j)) 1 >> $file4
	./main 8 $((2**j)) 0 >> $file4
	
	./main 512 $((2**j)) 1 >> $file5
	./main 512 $((2**j)) 0 >> $file5

	./main 8096 $((2**j)) 1 >> $file6
	./main 8096 $((2**j)) 0 >> $file6
done
