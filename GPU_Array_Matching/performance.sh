#!/bin/bash

# array_size goes from 3 to 8096
for i in {3..13}
do	
	./main $((2**i)) 256 1 >> change_arr_size.csv
	./main $((2**i)) 256 0 >> change_arr_size.csv
done

# num_arrays goes from 8 to 1024
for j in {3..10}
do		
	./main 64 $((2**j)) 1 >> change_num_arrays.csv
	./main 64 $((2**j)) 0 >> change_num_arrays.csv
done
