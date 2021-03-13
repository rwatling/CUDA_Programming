#!/bin/bash

# array_size
for i in {3..14}
do	
	# num_arrays
	for j in {3..10}
	do		
		# Run the same parameters 8 times
		for k in {1..8}
		do
			./main $((2**i)) $((2**j)) 1 >> results.csv
			./main $((2**i)) $((2**j)) 0 >> results.csv
		done
	done
done
