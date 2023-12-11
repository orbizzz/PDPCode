#pragma once

#include <vector>
#include <string>

using namespace std;
#define MAX_LENGTH (10)

#define OFFSET0  0
#define OFFSET1  OFFSET0  + 4345*256
#define OFFSET2  OFFSET1  + 256*256
#define OFFSET3  OFFSET2  + 256*256
#define OFFSET4  OFFSET3  + 256*256
#define OFFSET5  OFFSET4  + 256*256
#define OFFSET6  OFFSET5  + 256*256
#define OFFSET7  OFFSET6  + 256*256
#define OFFSET8  OFFSET7  + 256
#define OFFSET9  OFFSET8  + 256
#define OFFSET10 OFFSET9  + 256  
#define OFFSET11 OFFSET10 + 256
#define OFFSET12 OFFSET11 + 256
#define OFFSET13 OFFSET12 + 256
#define OFFSET14 OFFSET13 + 2803*256
#define OFFSET15 OFFSET14 + 256*256
#define OFFSET16 OFFSET15 + 256*256
#define OFFSET17 OFFSET16 + 256*256
#define OFFSET18 OFFSET17 + 256*256
#define OFFSET19 OFFSET18 + 256*256
#define OFFSET20 OFFSET19 + 256*256
#define OFFSET21 OFFSET20 + 256
#define OFFSET22 OFFSET21 + 256
#define OFFSET23 OFFSET22 + 256
#define OFFSET24 OFFSET23 + 256
#define OFFSET25 OFFSET24 + 256
#define OFFSET26 OFFSET25 + 256
#define OFFSET27 OFFSET26 + 10*512
#define OFFSET28 OFFSET27 + 10
#define OFFSET29 OFFSET28 + 256*512
#define OFFSET30 OFFSET29 + 256
#define OFFSET31 OFFSET30 + 2803*256
#define OFFSET32 OFFSET31 + 2803

struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  int num_elem();
  void fill_zeros();

  float *buf = nullptr;
  int ndim = 0;
  int shape[4];
};

void initialize_translator(const char *, int);
void finalize_translator();
void translator(Tensor *, Tensor *, int);
