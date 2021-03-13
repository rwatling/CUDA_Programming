#!/bin/bash

# array_size goes from 3 to 8096
for i in {3..13}
do	
	# num_arrays goes from 8 to 1024
	for j in {3..10}
	do		
		# Run the same parameters 4 times
		for k in {1..4}
		do
			./main $((2**i)) $((2**j)) 1 >> results.csv
			./main $((2**i)) $((2**j)) 0 >> results.csv
		done
	done
done
