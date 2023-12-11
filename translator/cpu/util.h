#pragma once

#include "translator.h"

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <cstring>

using namespace std;

class Lang{
public:
  Lang(string, string, string, int);
  ~Lang();
  map<string, int> word2index_;
  map<string, int> word2count_;
  map<int, string> index2word_;
  int n_words_;
};

void print_help();
void parse_option(int, char**, int *, bool *, bool *, bool *);
void load_input(Tensor *, int, bool, bool, bool);
void *read_binary(const char *, size_t *);
double get_time();
void write_output(Tensor *, const char *, int);
void save_translation_result(Tensor *, bool, int);
void check_validation(const char *, bool, int);
void warmup_translator(Tensor *, Tensor *, bool, int);
