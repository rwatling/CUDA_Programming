CC = nvcc
CFLAGS = -std=c++11
LDFLAGS = -lnvidia-ml
IFLAGS = -I../../
all: transpose

transpose: transpose.o
	$(CC) ${CFLAGS} ${IFLAGS} -o transpose transpose.o ${LDFLAGS}

transpose.o: transpose.cu
	$(CC) ${CFLAGS} ${IFLAGS} -c transpose.cu

clean:
	-rm -f *.o transpose
