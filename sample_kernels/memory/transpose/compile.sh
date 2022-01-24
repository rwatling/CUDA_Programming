#!/bin/bash

nvcc -I../../ transpose.cu -std=c++11 -o transpose -lnvidia-ml
