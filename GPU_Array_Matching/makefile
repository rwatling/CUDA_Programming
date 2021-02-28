NVCC = nvcc
CFLAGS =
OBJ = main.o rand_init.o
EXEC = main
LFLAGS=-lcurand

all: main

main: $(OBJ)
	$(NVCC) -o $(EXEC) $(OBJ) $(LFLAGS)

main.o: main.cu cuda_includes.h
	$(NVCC) $(CFLAGS) -c -o main.o main.cu

rand_init.o: rand_init.cu rand_init.h
	$(NVCC) $(CFLAGS) -c -o rand_init.o rand_init.cu
clean:
	rm $(EXEC)
	rm $(OBJ)
