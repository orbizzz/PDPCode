#!/bin/bash

# mpirun -np 1 ./translator_origin -n 4096 -v -w 2> result_cpu_origin
mpirun -np 1 ./translator -n 128 2>> results_128
