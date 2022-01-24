CC = nvcc
CFLAGS = -std=c++11
LDFLAGS = -lnvidia-ml
IFLAGS = -I../../
all: finite-difference

finite-difference: finite-difference.o
	$(CC) ${CFLAGS} ${IFLAGS} -o finite-difference finite-difference.o ${LDFLAGS}

finite-difference.o:
	$(CC) ${CFLAGS} ${IFLAGS} -c finite-difference.cu

clean:
	-rm -f *.o finite-difference
