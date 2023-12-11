#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 64
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif
void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
    int M_ = M / mpi_world_size;

    MPI_Bcast(B, K*N, MPI_FLOAT,
              0, MPI_COMM_WORLD);
    MPI_Scatter(A, M_*K, MPI_FLOAT,
                A, M_*K, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // #pragma omp parallel for num_threads(threads_per_process) shared(A, B, C) collapse(2)
    // for (int i = 0; i < M_; i += BLOCK_SIZE) {
    //     for (int j = 0; j < N; j += BLOCK_SIZE) {
    //         for (int k = 0; k < K; k += BLOCK_SIZE) {
    //             for (int ii = i; ii < min(M, i + BLOCK_SIZE); ii++) {
    //                 for (int kk = k; kk < min(K, k + BLOCK_SIZE); kk++) {
    //                     int JJ = min(N, j + BLOCK_SIZE);
    //                     #pragma omp simd
    //                     for (int jj = j; jj < JJ; jj++) {
    //                         C[ii*N + jj] += A[ii*K + kk] * B[kk*N + jj];
    //                     } 
    //                 } 
    //             } 
    //         } 
    //     } 
    // } 

    #pragma omp parallel for num_threads(threads_per_process) shared(A, B, C) collapse(2)
    for (int i = 0; i < M_; i += BLOCK_SIZE) {
        for (int k = 0; k < K; k += BLOCK_SIZE) {
            for (int ii = i; ii < min(M_, i + BLOCK_SIZE); ii++) {
                for (int kk = k; kk < min(K, k + BLOCK_SIZE); kk++) {
                    __m512 a = _mm512_set1_ps(A[ii*K + kk]);
                    #pragma omp simd
                    for (int jj = 0; jj < N; jj += 16) {
                        __m512 b = _mm512_loadu_ps(&B[kk*N + jj]);
                        __m512 c = _mm512_loadu_ps(&C[ii*N + jj]);
                        c = _mm512_fmadd_ps(a, b, c);
                        _mm512_storeu_ps(&C[ii*N + jj], c);
                    }
                }
            }
        }
    }

    MPI_Gather(C, M_*N, MPI_FLOAT,
            C, M_*N, MPI_FLOAT,
            0, MPI_COMM_WORLD);
} 
