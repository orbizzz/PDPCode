#!/bin/bash

mpirun -np 1 ./translator -n 4096 -v -w 2>> result_gpu 
mpirun -np 1 ./translator -n 16384 -v -w 2>> result_gpu 
