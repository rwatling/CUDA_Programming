#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#define MATRIX_SIZE 1024

static double a[MATRIX_SIZE][MATRIX_SIZE];
static double b[MATRIX_SIZE][MATRIX_SIZE];
static double c[MATRIX_SIZE][MATRIX_SIZE];

void matrix_multiply()
{
    double s;
    int i,j,k;

    for( i=0; i<MATRIX_SIZE; i++ ) {
        for( j=0; j<MATRIX_SIZE; j++ ) {
            a[i][j]=( double )i*( double )j;
            b[i][j]=( double )i/( double )( j+5 );
        }
    }

    for( j=0; j<MATRIX_SIZE; j++ ) {
        for( i=0; i<MATRIX_SIZE; i++ ) {
            s=0;
            for( k=0; k<MATRIX_SIZE; k++ ) {
                s+=a[i][k]*b[k][j];
            }
            c[i][j] = s;
        }
    }

    s=0.0;
    for( i=0; i<MATRIX_SIZE; i++ ) {
        for( j=0; j<MATRIX_SIZE; j++ ) {
            s+=c[i][j];
        }
    }
}

int main ( int argc, char **argv) {
    
    

    // Sample Work
    printf("Hello world!\n");

    matrix_multiply();
}