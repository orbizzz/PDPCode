#!/bin/bash

# mpirun -np 4 ./main_summa -n 10 -t 32 -v 4096 4096 4096 > result_summa
# mpirun -np 4 ./main -n 10 -t 32 -v 4096 4096 4096 > result

# avx512
mpirun -np 4 ./main_summa -n 10 -t 32 -v 8192 8192 8192 > result_summa_avx512
mpirun -np 4 ./main -n 10 -t 32 -v 8192 8192 8192 > result_avx512
