CC = nvcc
CFLAGS = -std=c++11
LDFLAGS = -lnvidia-ml
IFLAGS = -I../../
all: coalescing

coalescing: coalescing.o
	$(CC) ${CFLAGS} ${IFLAGS} -o coalescing coalescing.o ${LDFLAGS}

coalescing.o: coalescing.cu
	$(CC) ${CFLAGS} ${IFLAGS} -c coalescing.cu

clean:
	-rm -f *.o coalescing
