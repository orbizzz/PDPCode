#include "util.h"
#include "translator.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <mpi.h>

static const char parameter_fname[30] = "./data/parameters.bin";
static const char output_fname[30] = "./output.txt";

int main(int argc, char **argv) {

  int mpi_rank=0, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  
  int N = 1;
  bool V = false;
  bool S = false;
  bool W = false;
  parse_option(argc, argv, &N, &V, &S, &W);
  
  Tensor *input, *output;
  double st = 0.0, et = 0.0;
  input = new Tensor({N, MAX_LENGTH});
  output = new Tensor({N, MAX_LENGTH});

  load_input(input, N, V, S, W);
  
  initialize_translator(parameter_fname, N);
  warmup_translator(input, output, W, N); 
 
  if (mpi_rank == 0) {
    fprintf(stderr, " Translating %d sentence(s) ... ", N);
    st = get_time(); 
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  translator(input, output, N);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (mpi_rank == 0) {
    et = get_time();
    fprintf(stderr, "DONE!\n");
    write_output(output, output_fname, N);
    save_translation_result(output, S, N);
    fprintf(stderr, " ---------------------------------------------\n");
    fprintf(stderr, " Elapsed time : %lf s\n", et-st);
    fprintf(stderr, " Throughput   : %lf sentences/sec\n", (double)N/(et-st));
    check_validation(output_fname, V, N);
  }

  finalize_translator();
  MPI_Finalize();

  return EXIT_SUCCESS;
}
