#!/bin/sh

nvcc main.cu -O2 --ptxas-options=-v
