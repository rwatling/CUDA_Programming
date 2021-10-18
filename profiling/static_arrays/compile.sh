#!/bin/sh

nvcc main.cu -O3 --ptxas-options=-v
