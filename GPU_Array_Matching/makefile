CC = nvcc
CFLAGS =
OBJ = main.o

main: $(OBJ)
	$(CC) -o main $(OBJ)

main.o: main.cu
	$(CC) $(CFLAGS) -c -o main.o main.cu

clean:
	rm main
	rm $(OBJ)
