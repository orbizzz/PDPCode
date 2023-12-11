#include <getopt.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "matmul.h"
#include "util.h"

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8, K = 8;
static int threads_per_process = 1;
static int num_iterations = 1;
static int mpi_rank, mpi_world_size;

static void print_help(const char *prog_name) {
  printf(
      "Usage: %s [-pvh] [-t threads_per_process] [-n num_iterations] M N K\n",
      prog_name);
  printf("Options:\n");
  printf("     -p : print vector. (default: off)\n");
  printf("     -v : validate vector dot. (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -t : number of threads per process (default: 1)\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("      M : number of rows of matrix A and C. (default: 8)\n");
  printf("      N : number of columns of matrix B and C. (default: 8)\n");
  printf(
      "      K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 't':
        threads_per_process = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0:
        M = atoi(argv[i]);
        break;
      case 1:
        N = atoi(argv[i]);
        break;
      case 2:
        K = atoi(argv[i]);
        break;
      default:
        break;
    }
  }
  if (mpi_rank == 0) {
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of threads per process: %d\n", threads_per_process);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
  }
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("(%s) Hello world, rank %d out of %d\n", processor_name, mpi_rank,
      mpi_world_size);
  MPI_Barrier(MPI_COMM_WORLD);

  parse_opt(argc, argv);

  float *A, *B, *C;
  alloc_mat(&A, M, K);
  alloc_mat(&B, K, N);
  alloc_mat(&C, M, N);
  if (mpi_rank == 0) {
    printf("[rank %d] Initializing matrices...", mpi_rank);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    printf("Done!\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  ////
  float *block1, *block2;
  float *buf1, *buf2, *buf3;

  int q = (int)sqrt((double)mpi_world_size);

  int M_ = M/q;
  int N_ = N/q;
  int K_ = K/q;

  int dims[2];
  int periods[2];
  int coordinates[2];
	int free_coords[2];
  int my_grid_rank, grid_rank;
  MPI_Comm grid_comm;
  MPI_Comm row_comm;
	MPI_Comm col_comm;
  dims[0] = dims[1] = q;
  periods[0] = periods[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
  MPI_Comm_rank(grid_comm, &my_grid_rank);
  MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
  MPI_Cart_rank(grid_comm, coordinates, &grid_rank);

  assert(grid_rank == my_grid_rank);

  free_coords[0] = 0;
  free_coords[1] = 1;
  MPI_Cart_sub(grid_comm, free_coords, &row_comm);
  free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);


  alloc_mat(&block1, M_, K_);
  alloc_mat(&buf1, M_, K_);
  alloc_mat(&block2, K_, N_);
  alloc_mat(&buf2, K_, N_);
  alloc_mat(&buf3, M_, N_);

  //////

  double elapsed_time_sum = 0;
  for (int i = 0; i <= num_iterations; ++i) {
    if (mpi_rank == 0) {
      if (i != 0) {
        printf("[rank %d] Calculating...(iter=%d) ", mpi_rank, i);
        fflush(stdout);
      }
      zero_mat(C, M, N);
      zero_mat(block1, M_, K_);
      zero_mat(block2, K_, N_);
      zero_mat(buf3, M_, N_);
    } else {
      zero_mat(A, M, K);
      zero_mat(B, K, N);
      zero_mat(C, M, N);
      zero_mat(block1, M_, K_);
      zero_mat(block2, K_, N_);
      zero_mat(buf3, M_, N_);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // timer_start(0);

///////
    if (mpi_rank == 0) {
      for (int a=0; a<q; a++) {
        for (int b=0; b<q; b++) {
          if (a == 0 && b == 0) continue;

          for (int c=0; c<M_; c++) {
            for (int d=0; d<K_; d++) {
              block1[c*K_ + d] = A[(a*M_ + c)*K + (b*K_ + d)];
            }
          }
          for (int c=0; c<K_; c++) {
            for (int d=0; d<N_; d++) {
              block2[c*N_ + d] = B[(a*M_ + c)*N + (b*K_ + d)];
            }
          }
          int dest;
          int coords[] = {a, b};
          MPI_Cart_rank(grid_comm, coords, &dest);

          MPI_Send(block1, M_*K_, MPI_FLOAT, dest, 0, grid_comm);
          MPI_Send(block2, K_*N_, MPI_FLOAT, dest, 0, grid_comm);
        }
      }
      for (int c=0; c<M_; c++) {
        for (int d=0; d<K_; d++) {
          block1[c*K_ + d] = A[c*K + d];
        }
      }
      for (int c=0; c<K_; c++) {
        for (int d=0; d<N_; d++) {
          block2[c*N_ + d] = B[c*N + d];
        }
      }
    } else {
      MPI_Recv(block1, M_*K_, MPI_FLOAT, 0, 0, grid_comm, MPI_STATUS_IGNORE);
      MPI_Recv(block2, K_*N_, MPI_FLOAT, 0, 0, grid_comm, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    timer_start(0);


    for (int k=0; k<q; k++) {
      if (coordinates[1] == k) {
        memcpy(buf1, block1, M_*K_*sizeof(float));
      }
      MPI_Bcast(buf1, M_*K_, MPI_FLOAT, k, row_comm);
      if (coordinates[0] == k) {
        memcpy(buf2, block2, K_*N_*sizeof(float));
      }
      MPI_Bcast(buf2, N_*K_, MPI_FLOAT, k, col_comm);

      matmul(buf1, buf2, buf3, M_, N_, K_, threads_per_process, mpi_rank, mpi_world_size);
    }

///////


    // matmul(A, B, C, M, N, K, threads_per_process, mpi_rank, mpi_world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    // if sync, comment it
    double elapsed_time = timer_stop(0);

    if (mpi_rank == 0) {
      for (int a=0; a<M_; a++) {
        for (int b=0; b<N_; b++) {
          C[a*N+b] = buf3[a*N_+b];
        }
      }
      for (int a=0; a<q; a++) {
        for (int b=0; b<q; b++) {
          if (a == 0 && b == 0) continue;
          int source;
          int coords[2] = {a, b};
          MPI_Cart_rank(grid_comm, coords, &source);
          MPI_Recv(buf3, M_*N_, MPI_FLOAT, source, 0, grid_comm, MPI_STATUS_IGNORE);
        
          for (int c=0; c<M_; c++) {
            for (int d=0; d<N_; d++) {
              C[(a*M_+c)*N+(b*N_+d)] = buf3[c*N_+d];
            }
          }
        }
      }
    } else {
      MPI_Send(buf3, M_*N_, MPI_FLOAT, 0, 0, grid_comm);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    // if sync, uncomment it
    // double elapsed_time = timer_stop(0);

    if (mpi_rank == 0) {
      printf("%f sec\n", elapsed_time);
      if (i != 0)
        elapsed_time_sum += elapsed_time;
    }
  }

  if (mpi_rank == 0) {
    if (print_matrix) {
      printf("MATRIX A:\n");
      print_mat(A, M, K);
      printf("MATRIX B:\n");
      print_mat(B, K, N);
      printf("MATRIX C:\n");
      print_mat(C, M, N);
    }

    if (validation) {
      check_mat_mul(A, B, C, M, N, K);
    }

    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("[rank %d] Avg. time: %f sec\n", mpi_rank, elapsed_time_avg);
    printf("[rank %d] Avg. throughput: %f GFLOPS\n", mpi_rank,
        2.0 * M * N * K / elapsed_time_avg / 1e9);
  }

  MPI_Finalize();
  return 0;
}
