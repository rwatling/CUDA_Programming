NVCC = nvcc
CFLAGS =
OBJ = main.o array_match.o shm_array_match.o
EXEC = main
LFLAGS=-lcurand

all: main

main: $(OBJ)
	$(NVCC) -o $(EXEC) $(OBJ) $(LFLAGS)

main.o: main.cu cuda_includes.h
	$(NVCC) $(CFLAGS) -o main.o -c main.cu

array_match.o: array_match.cu array_match.h cuda_includes.h
	$(NVCC) $(CFLAGS) -o array_match.o -c array_match.cu

shm_array_match.o: shm_array_match.cu shm_array_match.h cuda_includes.h
	$(NVCC) $(CFLAGS) -o shm_array_match.o -c shm_array_match.cu

clean:
	rm $(EXEC)
	rm $(OBJ)
