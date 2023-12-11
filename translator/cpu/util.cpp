#include "util.h"
#include "translator.h"

#include <mpi.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <unistd.h>
#include <fstream>

using namespace std;

static map<string, string> fra_eng_pairs;

Lang::Lang(string filename1, string filename2, string filename3, int n_words) {

  fstream file1, file2, file3;
  string key, value, t, q;
  
  // open word2index file
  file1.open(filename1.c_str());
  while (file1 >> key) {
    file1 >> value;
    this->word2index_.insert(pair<string, int>(key, stoi(value)));
  }
  file1.close();
  
  // open word2count file
  file2.open(filename2.c_str());
  while (file2 >> key) {
    file2 >> value;
    this->word2count_.insert(pair<string, int>(key, stoi(value)));
  }
  file2.close();

  // open index2word file
  file3.open(filename3.c_str());
  while (file3 >> key) {
    file3 >> value;
    this->index2word_.insert(pair<int, string>(stoi(key), value));
  }
  file3.close();
}

Lang::~Lang(){}

Lang input_lang = Lang( "./data/input_lang_word2index.txt", 
                        "./data/input_lang_word2count.txt", 
                        "./data/input_lang_index2word.txt", 4345);

Lang output_lang = Lang("./data/output_lang_word2index.txt", 
                        "./data/output_lang_word2count.txt", 
                        "./data/output_lang_index2word.txt", 2803);


void load_input(Tensor *input, int N, bool V, bool S, bool W){
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stderr, " Loading sentences from pairs.csv ... ");
    
    for (int i=0; i<N; ++i){
      for (int j=0; j<MAX_LENGTH; ++j){
        input->buf[i * MAX_LENGTH + j] = 0.0;
      }
    }

    FILE *file = fopen("./data/pairs.csv", "r");
    if(!file){
      fprintf(stderr, "Could not open the file\n");
      return;
    }
   
    char one_line[1024];
    while(fgets(one_line, 1024, file)){
      string fra_string = strtok(one_line, ",");
      string eng_string = strtok(NULL, "\n");
      fra_eng_pairs.insert(pair<string, string>(fra_string, eng_string));
    }

    char **fra = (char**)malloc(N * sizeof(char*));
    char **eng = (char**)malloc(N * sizeof(char*));
    for (int i=0; i<N; ++i) {
      fra[i] = (char*)malloc(1024 * sizeof(char));
      eng[i] = (char*)malloc(1024 * sizeof(char));
    }
   
    auto it = fra_eng_pairs.begin();
    for (int i=0; i<N; ++i) {
      strcpy(fra[i], it->first.c_str());
      strcpy(eng[i], it->second.c_str());
      it = next(it);
      if (it == fra_eng_pairs.end()) {
        it = fra_eng_pairs.begin();
      }
    }
    fprintf(stderr, "DONE!\n");

    fprintf(stderr, " Tokenizing input French sentences ... ");
    int **fra_word2index = (int**)calloc(N, sizeof(int*));
    for (int i=0; i<N; ++i){
      fra_word2index[i] = (int*)calloc(MAX_LENGTH, sizeof(int));
    }
    
    for (int i=0; i<N; ++i) {
      char fra_one_sentence[1024];
      strcpy(fra_one_sentence, fra[i]);
      string one_word = strtok(fra_one_sentence, " ");
      auto idx = input_lang.word2index_.find(one_word);
      input->buf[i*MAX_LENGTH + 0] = idx->second;
      for (int j=1; j<MAX_LENGTH; ++j){
        char *y = strtok(NULL, " ");
        if (y == NULL) {
          input->buf[i*MAX_LENGTH + j] = 1;
          break;
        }
        idx = input_lang.word2index_.find(string(y));
        input->buf[i*MAX_LENGTH + j] = idx->second;
      }
    }
    fprintf(stderr, "DONE!\n");

    for (int i=0; i<N; ++i) {
      delete fra[i];
      delete eng[i];
    }
    delete fra;
    delete eng;

    fclose(file);
  }
}


void save_translation_result(Tensor *output, bool S, int N) {
  if (S == false) return;
  fprintf(stderr, " Writing generated_sentences.txt   ... ");
  FILE *output_fp = (FILE *)fopen("generated_sentences.txt", "w");

  fprintf(output_fp, "\n Generated sentences\n");
  fprintf(output_fp, " =============================================\n");
  fprintf(output_fp, " [-->] input sentence in French\n");
  fprintf(output_fp, " [===] target sentence in English\n");
  fprintf(output_fp, " [<--] generated sentence\n\n");

  auto it = fra_eng_pairs.begin();
  for (int i=0; i<N; ++i){
    fprintf(output_fp, "                [ sentence %d ] \n", i);
    fprintf(output_fp, " [-->] %s\n", it->first.c_str());
    fprintf(output_fp, " [===] %s\n", it->second.c_str());
    fprintf(output_fp, " [<--] ");
    for (int m=0; m<MAX_LENGTH; ++m){
      int index = (int)output->buf[i*MAX_LENGTH + m];
      // EOS_token
      if (index == 1) {
        fprintf(output_fp, "\n");
        break;
      }
      else {
        auto word = output_lang.index2word_.find(index);
        fprintf(output_fp, "%s ", word->second.c_str());
      }
    }
    it = next(it);
    if (it == fra_eng_pairs.end()) {
      it = fra_eng_pairs.begin();
    }
    fprintf(output_fp, "\n");
  }
  fclose(output_fp);
  fprintf(stderr, "DONE!\n");
}

void *read_binary(const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "[ERROR] Cannot open file \'%s\'\n", filename);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  
  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    fprintf(stderr, "[ERROR] Cannot read file \'%s\'\n", filename);
    exit(-1);
  }
  fclose(f);

  if (size != NULL) *size = (size_t)(size_ / 4);  // float
  return buf;
}

void write_output(Tensor *output, const char *filename, int N) {
  fprintf(stderr, " Writing output ... ");
  fflush(stdout);
  FILE *output_fp = (FILE *)fopen(filename, "w");
  for (int i = 0; i < N; i++) {
    for (int j=0; j<MAX_LENGTH; ++j) {
      fprintf(output_fp, "%04d", (int)output->buf[i * MAX_LENGTH + j]);
    }
    // fprintf(output_fp, "\n");
  }   
  fclose(output_fp);
  fprintf(stderr, "DONE!\n");
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_help() {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stderr, " Usage: ./translator [-n num_input_sentences] [-vpwh]\n");
    fprintf(stderr, " Options:\n");
    fprintf(stderr, "  -n : number of input sentences (default: 1)\n");
    fprintf(stderr, "  -v : validate translator.      (default: off, available input size: ~2525184)\n");
    fprintf(stderr, "  -s : save generated sentences (default: off)\n");
    fprintf(stderr, "  -w : enable warmup (default: off)\n");
    fprintf(stderr, "  -h : print this page.\n");
  }
}

void parse_option(int argc, char **argv, int *N, bool *V, bool *S, bool *W) {

  int opt;
  while((opt = getopt(argc, argv, "n:vswh")) != -1) {
    switch (opt) {
      case 'n': *N = atoi(optarg); break;
      case 'v': *V = true; break;
      case 's': *S = true; break;
      case 'w': *W = true; break;
      case 'h': print_help(); exit(-1); break;
      default: print_help(); exit(-1); break;
    }
  }

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stderr, "\n Model : Translator\n");
    fprintf(stderr, " French to English translation\n");
    fprintf(stderr, " =============================================\n");
    fprintf(stderr, " Number of sentences : %d\n", (*N));
    fprintf(stderr, " Warming up : %s\n", (*W)? "ON":"OFF");
    fprintf(stderr, " Validation : %s\n", (*V)? "ON":"OFF");
    fprintf(stderr, " Save generated sentences : %s\n", (*S)? "ON":"OFF");
    fprintf(stderr, " ---------------------------------------------\n");
  }
}

void check_validation(const char *fname, bool V, int N) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    int num_print_error = 0;
    bool is_wrong = false;
    char c_answer[4];
    char c_output[4];
    FILE *answer_fp = NULL;
    FILE *output_fp = NULL;

    if (V == false) return;
    fprintf(stderr, " Validation   : ");

    answer_fp = (FILE*)fopen("./data/answer", "r");
    if (answer_fp == NULL) {
      fprintf(stderr, "Could not open data file\n");
      exit(-1);
    }
    
    output_fp = (FILE*)fopen(fname, "r");
    if (output_fp == NULL) {
      fprintf(stderr, "Could not open data file\n");
      exit(-1);
    }

    for (int i=0; i<N*MAX_LENGTH; ++i) {
      if (num_print_error == 10) break;
      bool is_match = true;

      for (int j=0; j<4; ++j) {
        c_answer[j] = getc(answer_fp);
        c_output[j] = getc(output_fp);
        if (c_answer[j] != c_output[j]) {
          if (is_wrong == false) {
            fprintf(stderr, "FAILED!\n");
            fprintf(stderr, " ---------------------------------------------\n");
            fprintf(stderr, " Print only a portion of the incorrect results.\n"); 
          }
          is_match = false;
          is_wrong = true;
        }
      }
      if (is_match != true) {
        num_print_error++;
        fprintf(stderr," [%d-th sentence, %d-th word] answer : %c%c%c%c <-> output : %c%c%c%c\n", \
                i / MAX_LENGTH, i % MAX_LENGTH, \
                c_answer[0], c_answer[1], c_answer[2], c_answer[3], \
                c_output[0], c_output[1], c_output[2], c_output[3]);
      }
    }

    if (!is_wrong) fprintf(stderr, "PASSED!\n");

    fclose(answer_fp);
    fclose(output_fp);
  }
}

void warmup_translator(Tensor *input, Tensor *output, bool W, int N){
  if (W == true) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) fprintf(stderr, " Warming up ... ");
    for (int i=0; i<3 && W==true; ++i){
      translator(input, output, N);
    }
    if (mpi_rank == 0) fprintf(stderr, "DONE!\n");
  }
  else return;
}
