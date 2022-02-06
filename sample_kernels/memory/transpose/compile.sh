#!/bin/bash

rm -rf transpose
nvcc -I../../ transpose.cu -std=c++11 -o transpose -lnvidia-ml
