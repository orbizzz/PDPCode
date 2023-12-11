#include "translator.h"
#include "util.h"
#include <mpi.h>
#include <math.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <omp.h>
#include <immintrin.h>

#define NUM_GPU 4
#define NUM_STREAM 4
#define TILE_SIZE 16

#define SOS_token 0
#define EOS_token 1
#define HIDDEN_SIZE 256
#define INPUT_VOCAB_SIZE 4345
#define OUTPUT_VOCAB_SIZE 2803
#define TAG 10000

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

int mpi_rank;
int mpi_size;
int num_devices;
int num_per_node = 0; // node 별 담당할 단어 개수(init에서 계산)
int num_per_gpu = 0; // GPU 별 담당할 단어 개수(init에서 계산)
int num_per_stream = 0;

cudaStream_t s0[NUM_GPU];
cudaStream_t s1[NUM_GPU][NUM_STREAM];

// encoder
float *d_ew_emb[NUM_GPU], *d_ew_ir[NUM_GPU], *d_ew_iz[NUM_GPU], *d_ew_in[NUM_GPU];
float *d_ew_hr[NUM_GPU], *d_ew_hz[NUM_GPU], *d_ew_hn[NUM_GPU];
float *d_eb_ir[NUM_GPU], *d_eb_iz[NUM_GPU], *d_eb_in[NUM_GPU];
float *d_eb_hr[NUM_GPU], *d_eb_hz[NUM_GPU], *d_eb_hn[NUM_GPU];

float *d_input[NUM_GPU], *d_encoder_embedded[NUM_GPU], *d_encoder_hidden[NUM_GPU], *d_encoder_rt[NUM_GPU];
float *d_encoder_zt[NUM_GPU], *d_encoder_nt[NUM_GPU], *d_encoder_outputs[NUM_GPU];

// decoder
float *d_dw_emb[NUM_GPU], *d_dw_ir[NUM_GPU], *d_dw_iz[NUM_GPU], *d_dw_in[NUM_GPU];
float *d_dw_hr[NUM_GPU], *d_dw_hz[NUM_GPU], *d_dw_hn[NUM_GPU];
float *d_db_ir[NUM_GPU], *d_db_iz[NUM_GPU], *d_db_in[NUM_GPU];
float *d_db_hr[NUM_GPU], *d_db_hz[NUM_GPU], *d_db_hn[NUM_GPU];
float *d_dw_attn[NUM_GPU], *d_db_attn[NUM_GPU], *d_dw_attn_comb[NUM_GPU], *d_db_attn_comb[NUM_GPU];
float *d_dw_out[NUM_GPU], *d_db_out[NUM_GPU];

float *d_decoder_input[NUM_GPU], *d_decoder_embedded[NUM_GPU], *d_decoder_attn_weights[NUM_GPU], *d_decoder_attn_linear[NUM_GPU];
float *d_decoder_attn_applied[NUM_GPU], *d_decoder_relu[NUM_GPU], *d_decoder_rt[NUM_GPU];
float *d_decoder_zt[NUM_GPU], *d_decoder_nt[NUM_GPU], *d_decoder_fin[NUM_GPU], *d_output[NUM_GPU];

// enable
int *d_enable[NUM_GPU];

Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i=0; i<ndim; ++i) { 
    shape[i] = shape_[i]; 
  }
  int N_ = num_elem();
  buf = (float *)calloc(N_, sizeof(float));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i=0; i<ndim; ++i) { 
    shape[i] = shape_[i]; 
  }
  int N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  for (int n = 0; n < N_; ++n) {
    buf[n] = buf_[n]; 
  }
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

int Tensor::num_elem() {
  int sz = 1;
  for (int i=0; i<ndim; ++i){
    sz *= shape[i];
  }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n=0; n<N_; ++n) { 
    buf[n] = 0.0; 
  }
}

// Parameters
Tensor *eW_emb;
Tensor *eW_ir, *eW_iz, *eW_in;
Tensor *eW_hr, *eW_hz, *eW_hn;
Tensor *eb_ir, *eb_iz, *eb_in;
Tensor *eb_hr, *eb_hz, *eb_hn;
Tensor *dW_emb;
Tensor *dW_ir, *dW_iz, *dW_in;
Tensor *dW_hr, *dW_hz, *dW_hn;
Tensor *db_ir, *db_iz, *db_in;
Tensor *db_hr, *db_hz, *db_hn;
Tensor *dW_attn, *db_attn, *dW_attn_comb, *db_attn_comb, *dW_out, *db_out;

__global__ void d_embedding(int i, const float *input, const float *weight, float *output, int* dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (dev[row] == 0) {
    int aa = input[row * MAX_LENGTH + i];
    output[row * HIDDEN_SIZE + col] = weight[aa * HIDDEN_SIZE + col];
  }
}

//
__global__ void d_embedding_en(int i, const float *input, const float *weight, float *output, int* dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (input[row * MAX_LENGTH + i] == 0.0) dev[row] = 1;
  if (dev[row] == 0) {
    int aa = input[row * MAX_LENGTH + i];
    output[row * HIDDEN_SIZE + col] = weight[aa * HIDDEN_SIZE + col];
  }
}

__global__ void d_wandb_improved(const float *input1, const float *input2, const float *weight1, const float *weight2, const float *bias1, const float *bias2, float *output, int *dev) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x); // num
  int col = (blockIdx.y * blockDim.y + threadIdx.y); // HIDDEN_SIZE

  __shared__ float sA1[TILE_SIZE][TILE_SIZE];
  __shared__ float sA2[TILE_SIZE][TILE_SIZE];
  __shared__ float sB1[TILE_SIZE][TILE_SIZE];
  __shared__ float sB2[TILE_SIZE][TILE_SIZE];

  float tmp = 0.0;

  for (int k=0; k < (HIDDEN_SIZE / TILE_SIZE); ++k) {
    sA1[threadIdx.x][threadIdx.y] = input1[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE];
    sA2[threadIdx.x][threadIdx.y] = input2[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE];
    sB1[threadIdx.y][threadIdx.x] = weight1[(threadIdx.x + k * TILE_SIZE) + col * HIDDEN_SIZE];
    sB2[threadIdx.y][threadIdx.x] = weight2[(threadIdx.x + k * TILE_SIZE) + col * HIDDEN_SIZE];


    __syncthreads();

    for (int j=0; j<TILE_SIZE; ++j) {
      tmp += sB1[threadIdx.y][j] * sA1[threadIdx.x][j] + sB2[threadIdx.y][j] * sA2[threadIdx.x][j];
    }
    __syncthreads();
  }

  if (dev[row] == 0)  { 
    tmp = tmp + bias1[col] + bias2[col];

    output[row * HIDDEN_SIZE + col] = 1.0 / (1.0 + expf(-tmp));
  }
}
//

__global__ void d_wandb_improved_mul(const float *input1, const float *input2, const float *input3, const float *weight1, const float *weight2, const float *bias1, const float *bias2, float *output, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; // num
  int col = blockIdx.y * blockDim.y + threadIdx.y; // HIDDEN_SIZE

  __shared__ float sA1[TILE_SIZE][TILE_SIZE];
  __shared__ float sA2[TILE_SIZE][TILE_SIZE];
  __shared__ float sB1[TILE_SIZE][TILE_SIZE];
  __shared__ float sB2[TILE_SIZE][TILE_SIZE];

  float tmp1 = 0.0;
  float tmp2 = 0.0;

  for (int k=0; k < (HIDDEN_SIZE / TILE_SIZE); ++k) {
    sA1[threadIdx.x][threadIdx.y] = input1[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE];
    sA2[threadIdx.x][threadIdx.y] = input2[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE];
    sB1[threadIdx.y][threadIdx.x] = weight1[(threadIdx.x + k * TILE_SIZE) + col * HIDDEN_SIZE];
    sB2[threadIdx.y][threadIdx.x] = weight2[(threadIdx.x + k * TILE_SIZE) + col * HIDDEN_SIZE];

    __syncthreads();

    for (int j=0; j<TILE_SIZE; ++j) {
      tmp1 += sB1[threadIdx.y][j] * sA1[threadIdx.x][j];
      tmp2 += sB2[threadIdx.y][j] * sA2[threadIdx.x][j];
    }
    __syncthreads();
  }

  if (dev[row] == 0)  { 
    tmp1 += bias1[col] + (tmp2 + bias2[col]) * input3[row * HIDDEN_SIZE + col];

    output[row * HIDDEN_SIZE + col] =  tanhf(tmp1);
  }
}


__global__ void d_copy_encoder_outputs_elem(float *input1, float *input2, float *hidden, float *output, int i, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

  if (dev[row] == 0) {
    float4 x1 = FETCH_FLOAT4(input1[row * HIDDEN_SIZE + col]);
    float4 x2 = FETCH_FLOAT4(input2[row * HIDDEN_SIZE + col]);
    float4 x3 = FETCH_FLOAT4(hidden[row * HIDDEN_SIZE + col]);
    float4 x4;
    x4.x = (1 - x1.x) * x2.x + x1.x * x3.x;
    x4.y = (1 - x1.y) * x2.y + x1.y * x3.y;
    x4.z = (1 - x1.z) * x2.z + x1.z * x3.z;
    x4.w = (1 - x1.w) * x2.w + x1.w * x3.w;

    FETCH_FLOAT4(hidden[row * HIDDEN_SIZE + col]) =  x4;
    FETCH_FLOAT4(output[row * MAX_LENGTH * HIDDEN_SIZE + i * HIDDEN_SIZE + col]) = x4;
  }
}

__global__ void d_concat_linear_relu(const float *input1, const float *input2, const float *weight, const float *bias, float *output, int *dev) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x); // num
  int col = (blockIdx.y * blockDim.y + threadIdx.y); // HIDDEN_SIZE

  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  float tmp = 0.0;

  for (int k=0; k < HIDDEN_SIZE * 2 / TILE_SIZE; ++k) {
    if (k < HIDDEN_SIZE / TILE_SIZE) {
      sA[threadIdx.x][threadIdx.y] = input1[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE];
    } else {
      sA[threadIdx.x][threadIdx.y] = input2[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE - HIDDEN_SIZE];
    }
    sB[threadIdx.y][threadIdx.x] = weight[(k * TILE_SIZE + threadIdx.x) + col *2* HIDDEN_SIZE];
  
    __syncthreads();

    for (int j=0; j<TILE_SIZE; ++j) {
      tmp += sA[threadIdx.x][j] * sB[threadIdx.y][j];
    }

    __syncthreads();
  }

  tmp += bias[col];
  if (dev[row] == 0)  { 
    output[row * HIDDEN_SIZE + col] = (tmp < 0.0) ? 0.0 : tmp;
  }
}


__global__ void d_concat_linear(const float *input1, const float *input2,const float *weight, const float *bias, float *output, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (dev[row] == 0) {
    float c = bias[col];
    for (int k=0; k<HIDDEN_SIZE; ++k) {
      c += input1[row * HIDDEN_SIZE + k] * weight[col*2*HIDDEN_SIZE + k];
    }
    for (int k=HIDDEN_SIZE; k<2*HIDDEN_SIZE; ++k) {
      c += input2[row * HIDDEN_SIZE + k - HIDDEN_SIZE] * weight[col*2*HIDDEN_SIZE + k];
    }
    output[row * MAX_LENGTH + col] = c;
  }
}

__global__ void d_softmax(const float *input, float *output, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (dev[row] == 0) {
    float sum = 0.0;
    for (int m=0; m < MAX_LENGTH; ++m) {
      sum += expf(input[row * MAX_LENGTH + m]);
    }

    for (int m=0; m < MAX_LENGTH; ++m) {
      output[row * MAX_LENGTH + m] = expf(input[row * MAX_LENGTH + m]) / sum;
    }
  }
}

__global__ void d_bmm(const float *input, const float *weight, float *output, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (dev[row] == 0) {
    float c = 0.0;
    for (int k=0; k<MAX_LENGTH; ++k) {
      c += input[row * MAX_LENGTH + k] * weight[row * MAX_LENGTH * HIDDEN_SIZE + k * HIDDEN_SIZE + col]; 
    }
    output[row * HIDDEN_SIZE + col] = c;
  }
}


__global__ void d_elemwise_op(float *input1, float *input2, float *hidden, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

  if (dev[row] == 0) {
    float4 x1 = FETCH_FLOAT4(input1[row * HIDDEN_SIZE + col]);
    float4 x2 = FETCH_FLOAT4(input2[row * HIDDEN_SIZE + col]);
    float4 x3 = FETCH_FLOAT4(hidden[row * HIDDEN_SIZE + col]);
    float4 x4;
    x4.x = (1 - x1.x) * x2.x + x1.x * x3.x;
    x4.y = (1 - x1.y) * x2.y + x1.y * x3.y;
    x4.z = (1 - x1.z) * x2.z + x1.z * x3.z;
    x4.w = (1 - x1.w) * x2.w + x1.w * x3.w;

    FETCH_FLOAT4(hidden[row * HIDDEN_SIZE + col]) =  x4;
  }
}


__global__ void d_wandb_fin(float *hidden, float *weight, float *bias, float *output, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  float tmp = 0.0;

  for (int k=0; k < (HIDDEN_SIZE / TILE_SIZE); ++k) {
    sA[threadIdx.x][threadIdx.y] = hidden[row * HIDDEN_SIZE + threadIdx.y + k * TILE_SIZE];
    if (col < OUTPUT_VOCAB_SIZE) {
      sB[threadIdx.y][threadIdx.x] = weight[(threadIdx.x + k * TILE_SIZE) + col * HIDDEN_SIZE];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int j=0; j<TILE_SIZE; ++j) {
      tmp += sB[threadIdx.y][j] * sA[threadIdx.x][j];
    }
    __syncthreads();
  }

  if (dev[row] == 0 && col < OUTPUT_VOCAB_SIZE)  { 
    output[row * OUTPUT_VOCAB_SIZE + col] =  tmp + bias[col];
  }
}

__global__ void d_top_one_soft(float *hidden, float *output, float *origin, int i, int *dev) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (dev[row] == 0) {
    // float sum = 0.0;
    // for (int m=0; m<OUTPUT_VOCAB_SIZE; ++m) {
    //   sum += expf(hidden[row * OUTPUT_VOCAB_SIZE + m]);
    // }
    // int topi = 0 ;
    // float topval = logf(expf(hidden[row * OUTPUT_VOCAB_SIZE + 0]) / sum);
    // for (int m=1; m<OUTPUT_VOCAB_SIZE; ++m) {
    //   float x = logf(expf(hidden[row * OUTPUT_VOCAB_SIZE + m])  / sum);
    //   if (x >= topval) {
    //     topi = m;
    //     topval = x;
    //   }
    // }

    int topi = 0 ;
    float topval = (hidden[row * OUTPUT_VOCAB_SIZE + 0]);
    for (int m=1; m<OUTPUT_VOCAB_SIZE; ++m) {
      float x = hidden[row * OUTPUT_VOCAB_SIZE + m];
      if (x >= topval) {
        topi = m;
        topval = x;
      }
    }
    if (topi == EOS_token) {
      output[row * MAX_LENGTH + i] = EOS_token;
      dev[row] = 1;
    } else {
      output[row * MAX_LENGTH + i] = topi;
      if (i + 1 < MAX_LENGTH) origin[row * MAX_LENGTH + (i+1)] = topi;
    }
  }
}

void translator(Tensor *input, Tensor *output, int N){
  MPI_Request reqs1[13*3];
  MPI_Status status1[13*3];
  MPI_Request reqs2[19*3];
  MPI_Status status2[19*3];
  if (mpi_rank == 0) {
    for (int r=1; r < mpi_size; r++) {
      MPI_Isend((void *)(eW_emb->buf), INPUT_VOCAB_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+1, MPI_COMM_WORLD, &reqs1[(r-1)*13+0]);
      MPI_Isend((void *)(eW_ir->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+2, MPI_COMM_WORLD, &reqs1[(r-1)*13+1]);
      MPI_Isend((void *)(eW_iz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+3, MPI_COMM_WORLD, &reqs1[(r-1)*13+2]);
      MPI_Isend((void *)(eW_in->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+4, MPI_COMM_WORLD, &reqs1[(r-1)*13+3]);
      MPI_Isend((void *)(eW_hr->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+5, MPI_COMM_WORLD, &reqs1[(r-1)*13+4]);
      MPI_Isend((void *)(eW_hz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+6, MPI_COMM_WORLD, &reqs1[(r-1)*13+5]);
      MPI_Isend((void *)(eW_hn->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+7, MPI_COMM_WORLD, &reqs1[(r-1)*13+6]);
      MPI_Isend((void *)(eb_ir->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+8, MPI_COMM_WORLD, &reqs1[(r-1)*13+7]);
      MPI_Isend((void *)(eb_iz->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+9, MPI_COMM_WORLD, &reqs1[(r-1)*13+8]);
      MPI_Isend((void *)(eb_in->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+10, MPI_COMM_WORLD, &reqs1[(r-1)*13+9]);
      MPI_Isend((void *)(eb_hr->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+11, MPI_COMM_WORLD, &reqs1[(r-1)*13+10]);
      MPI_Isend((void *)(eb_hz->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+12, MPI_COMM_WORLD, &reqs1[(r-1)*13+11]);
      MPI_Isend((void *)(eb_hn->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+13, MPI_COMM_WORLD, &reqs1[(r-1)*13+12]);
      MPI_Isend((void *)(dW_emb->buf), OUTPUT_VOCAB_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+14, MPI_COMM_WORLD, &reqs2[(r-1)*19+0]);
      MPI_Isend((void *)(dW_ir->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+15, MPI_COMM_WORLD, &reqs2[(r-1)*19+1]);
      MPI_Isend((void *)(dW_iz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+16, MPI_COMM_WORLD, &reqs2[(r-1)*19+2]);
      MPI_Isend((void *)(dW_in->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+17, MPI_COMM_WORLD, &reqs2[(r-1)*19+3]);
      MPI_Isend((void *)(dW_hr->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+18, MPI_COMM_WORLD, &reqs2[(r-1)*19+4]);
      MPI_Isend((void *)(dW_hz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+19, MPI_COMM_WORLD, &reqs2[(r-1)*19+5]);
      MPI_Isend((void *)(dW_hn->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+20, MPI_COMM_WORLD, &reqs2[(r-1)*19+6]);
      MPI_Isend((void *)(db_ir->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+21, MPI_COMM_WORLD, &reqs2[(r-1)*19+7]);
      MPI_Isend((void *)(db_iz->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+22, MPI_COMM_WORLD, &reqs2[(r-1)*19+8]);
      MPI_Isend((void *)(db_in->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+23, MPI_COMM_WORLD, &reqs2[(r-1)*19+9]);
      MPI_Isend((void *)(db_hr->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+24, MPI_COMM_WORLD, &reqs2[(r-1)*19+10]);
      MPI_Isend((void *)(db_hz->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+25, MPI_COMM_WORLD, &reqs2[(r-1)*19+11]);
      MPI_Isend((void *)(db_hn->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+26, MPI_COMM_WORLD, &reqs2[(r-1)*19+12]);
      MPI_Isend((void *)(dW_attn->buf), MAX_LENGTH * 2 * HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+27, MPI_COMM_WORLD, &reqs2[(r-1)*19+13]);
      MPI_Isend((void *)(db_attn->buf), MAX_LENGTH, MPI_FLOAT, r, TAG+100*r+28, MPI_COMM_WORLD, &reqs2[(r-1)*19+14]);
      MPI_Isend((void *)(dW_attn_comb->buf), HIDDEN_SIZE * 2 * HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+29, MPI_COMM_WORLD, &reqs2[(r-1)*19+15]);
      MPI_Isend((void *)(db_attn_comb->buf), HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+30, MPI_COMM_WORLD, &reqs2[(r-1)*19+16]);
      MPI_Isend((void *)(dW_out->buf), OUTPUT_VOCAB_SIZE * HIDDEN_SIZE, MPI_FLOAT, r, TAG+100*r+31, MPI_COMM_WORLD, &reqs2[(r-1)*19+17]);
      MPI_Isend((void *)(db_out->buf), OUTPUT_VOCAB_SIZE, MPI_FLOAT, r, TAG+100*r+32, MPI_COMM_WORLD, &reqs2[(r-1)*19+18]);
    }
  } else {
    MPI_Irecv((void *)(eW_emb->buf), INPUT_VOCAB_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+1, MPI_COMM_WORLD, &reqs1[0]);
    MPI_Irecv((void *)(eW_ir->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+2, MPI_COMM_WORLD, &reqs1[1]);
    MPI_Irecv((void *)(eW_iz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+3, MPI_COMM_WORLD, &reqs1[2]);
    MPI_Irecv((void *)(eW_in->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+4, MPI_COMM_WORLD, &reqs1[3]);
    MPI_Irecv((void *)(eW_hr->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+5, MPI_COMM_WORLD, &reqs1[4]);
    MPI_Irecv((void *)(eW_hz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+6, MPI_COMM_WORLD, &reqs1[5]);
    MPI_Irecv((void *)(eW_hn->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+7, MPI_COMM_WORLD, &reqs1[6]);
    MPI_Irecv((void *)(eb_ir->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+8, MPI_COMM_WORLD, &reqs1[7]);
    MPI_Irecv((void *)(eb_iz->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+9, MPI_COMM_WORLD, &reqs1[8]);
    MPI_Irecv((void *)(eb_in->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+10, MPI_COMM_WORLD, &reqs1[9]);
    MPI_Irecv((void *)(eb_hr->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+11, MPI_COMM_WORLD, &reqs1[10]);
    MPI_Irecv((void *)(eb_hz->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+12, MPI_COMM_WORLD, &reqs1[11]);
    MPI_Irecv((void *)(eb_hn->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+13, MPI_COMM_WORLD, &reqs1[12]);
    MPI_Irecv((void *)(dW_emb->buf), OUTPUT_VOCAB_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+14, MPI_COMM_WORLD, &reqs2[0]);
    MPI_Irecv((void *)(dW_ir->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+15, MPI_COMM_WORLD, &reqs2[1]);
    MPI_Irecv((void *)(dW_iz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+16, MPI_COMM_WORLD, &reqs2[2]);
    MPI_Irecv((void *)(dW_in->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+17, MPI_COMM_WORLD, &reqs2[3]);
    MPI_Irecv((void *)(dW_hr->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+18, MPI_COMM_WORLD, &reqs2[4]);
    MPI_Irecv((void *)(dW_hz->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+19, MPI_COMM_WORLD, &reqs2[5]);
    MPI_Irecv((void *)(dW_hn->buf), HIDDEN_SIZE*HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+20, MPI_COMM_WORLD, &reqs2[6]);
    MPI_Irecv((void *)(db_ir->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+21, MPI_COMM_WORLD, &reqs2[7]);
    MPI_Irecv((void *)(db_iz->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+22, MPI_COMM_WORLD, &reqs2[8]);
    MPI_Irecv((void *)(db_in->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+23, MPI_COMM_WORLD, &reqs2[9]);
    MPI_Irecv((void *)(db_hr->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+24, MPI_COMM_WORLD, &reqs2[10]);
    MPI_Irecv((void *)(db_hz->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+25, MPI_COMM_WORLD, &reqs2[11]);
    MPI_Irecv((void *)(db_hn->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+26, MPI_COMM_WORLD, &reqs2[12]);
    MPI_Irecv((void *)(dW_attn->buf), MAX_LENGTH * 2 * HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+27, MPI_COMM_WORLD, &reqs2[13]);
    MPI_Irecv((void *)(db_attn->buf), MAX_LENGTH, MPI_FLOAT, 0, TAG+100*mpi_rank+28, MPI_COMM_WORLD, &reqs2[14]);
    MPI_Irecv((void *)(dW_attn_comb->buf), HIDDEN_SIZE * 2 * HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+29, MPI_COMM_WORLD, &reqs2[15]);
    MPI_Irecv((void *)(db_attn_comb->buf), HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+30, MPI_COMM_WORLD, &reqs2[16]);
    MPI_Irecv((void *)(dW_out->buf), OUTPUT_VOCAB_SIZE * HIDDEN_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+31, MPI_COMM_WORLD, &reqs2[17]);
    MPI_Irecv((void *)(db_out->buf), OUTPUT_VOCAB_SIZE, MPI_FLOAT, 0, TAG+100*mpi_rank+32, MPI_COMM_WORLD, &reqs2[18]);
  }
  MPI_Scatter((void *)(input->buf), num_per_node * MAX_LENGTH, MPI_FLOAT, (void *)(input->buf), num_per_node * MAX_LENGTH, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  if (mpi_rank == 0) {
    for (int r = 1; r < mpi_size; r++) {
      for (int i = 0; i < 13; i++) {
        MPI_Wait(&reqs1[(r-1)*13+i], &status1[(r-1)*13+i]);
      }
      for (int i = 0; i < 19; i++) {
        MPI_Wait(&reqs2[(r-1)*19+i], &status2[(r-1)*19+i]);
      }
    }
  } else {
    for (int i = 0; i < 13; i++) {
      MPI_Wait(&reqs1[i], &status1[i]);
    }
    for (int i = 0; i < 19; i++) {
      MPI_Wait(&reqs2[i], &status2[i]);
    }
  }


  #pragma omp parallel for num_threads(num_devices)
  for (int i=0; i < num_devices; i++) {
    cudaSetDevice(i);
    cudaMemcpyAsync(d_ew_emb[i], eW_emb->buf, INPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_ir[i], eW_ir->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_iz[i], eW_iz->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_in[i], eW_in->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_hr[i], eW_hr->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_hz[i], eW_hz->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_hn[i], eW_hn->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_eb_ir[i], eb_ir->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_eb_iz[i], eb_iz->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_eb_in[i], eb_in->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_eb_hr[i], eb_hr->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_eb_hz[i], eb_hz->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_eb_hn[i], eb_hn->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_ew_emb[i], eW_emb->buf, INPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_emb[i], dW_emb->buf, OUTPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_ir[i], dW_ir->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_iz[i], dW_iz->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_in[i], dW_in->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_hr[i], dW_hr->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_hz[i], dW_hz->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_hn[i], dW_hn->buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_ir[i], db_ir->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_iz[i], db_iz->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_in[i], db_in->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_hr[i], db_hr->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_hz[i], db_hz->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_hn[i], db_hn->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_attn[i], dW_attn->buf, MAX_LENGTH * 2 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_attn[i], db_attn->buf, MAX_LENGTH * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_attn_comb[i], dW_attn_comb->buf, HIDDEN_SIZE * 2 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_attn_comb[i], db_attn_comb->buf, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_dw_out[i], dW_out->buf, OUTPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    cudaMemcpyAsync(d_db_out[i], db_out->buf, OUTPUT_VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice, s0[i]);
    
    cudaStreamSynchronize(s0[i]);

    #pragma omp parallel for num_threads(NUM_STREAM)
    for (int s=0; s<NUM_STREAM; ++s) {

      cudaMemsetAsync(d_encoder_hidden[i] + s * num_per_stream * HIDDEN_SIZE, 0, num_per_stream * HIDDEN_SIZE * sizeof(float), s1[i][s]);
      cudaMemsetAsync(d_encoder_outputs[i] + s * num_per_stream * MAX_LENGTH * HIDDEN_SIZE, 0, num_per_gpu * MAX_LENGTH * HIDDEN_SIZE * sizeof(float), s1[i][s]);
      cudaMemsetAsync(d_enable[i] + s * num_per_stream, 0, num_per_stream * sizeof(int), s1[i][s]);
      cudaMemcpyAsync(d_input[i] + s * num_per_stream * MAX_LENGTH, (input->buf) + (i * num_per_gpu + s * num_per_stream)* MAX_LENGTH, num_per_stream * MAX_LENGTH * sizeof(float), cudaMemcpyHostToDevice, s1[i][s]);

      for (int j = 0; j < MAX_LENGTH; ++j) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(num_per_stream / TILE_SIZE, HIDDEN_SIZE / TILE_SIZE);
        d_embedding_en<<<grid, block, 0, s1[i][s]>>>(j, d_input[i] + s * num_per_stream * MAX_LENGTH
                                                      , d_ew_emb[i], d_encoder_embedded[i] + s * num_per_stream * HIDDEN_SIZE
                                                      , d_enable[i] + s * num_per_stream);
        d_wandb_improved<<<grid, block, 0, s1[i][s]>>>(d_encoder_embedded[i] + s * num_per_stream * HIDDEN_SIZE
                                                     , d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                     , d_ew_ir[i], d_ew_hr[i], d_eb_ir[i], d_eb_hr[i]
                                                     , d_encoder_rt[i] + s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        d_wandb_improved<<<grid, block, 0, s1[i][s]>>>(d_encoder_embedded[i]+ s * num_per_stream * HIDDEN_SIZE
                                                     , d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                     , d_ew_iz[i], d_ew_hz[i], d_eb_iz[i], d_eb_hz[i]
                                                     , d_encoder_zt[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        d_wandb_improved_mul<<<grid, block, 0, s1[i][s]>>>(d_encoder_embedded[i]+ s * num_per_stream * HIDDEN_SIZE,
                                                         d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                         , d_encoder_rt[i]+ s * num_per_stream * HIDDEN_SIZE,
                                                          d_ew_in[i], d_ew_hn[i], d_eb_in[i], d_eb_hn[i],
                                                           d_encoder_nt[i]+ s * num_per_stream * HIDDEN_SIZE, 
                                                           d_enable[i]+ s * num_per_stream);
        dim3 block1(TILE_SIZE, TILE_SIZE);
        dim3 grid1(num_per_stream / TILE_SIZE, HIDDEN_SIZE /4 / TILE_SIZE);
        d_copy_encoder_outputs_elem<<<grid1, block1, 0, s1[i][s]>>>(d_encoder_zt[i]+ s * num_per_stream * HIDDEN_SIZE
                                                          , d_encoder_nt[i]+ s * num_per_stream * HIDDEN_SIZE
                                                          , d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                          , d_encoder_outputs[i]+ s * num_per_stream * MAX_LENGTH * HIDDEN_SIZE, j, 
                                                          d_enable[i]+ s * num_per_stream);
      }

      cudaMemsetAsync(d_enable[i] + s * num_per_stream, 0, num_per_stream * sizeof(int), s1[i][s]);
      cudaMemsetAsync(d_decoder_input[i] + s * num_per_stream * MAX_LENGTH, SOS_token, num_per_stream * MAX_LENGTH * sizeof(float), s1[i][s]);
      
      for (int j = 0; j < MAX_LENGTH; ++j) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(num_per_stream / TILE_SIZE, HIDDEN_SIZE / TILE_SIZE);

        // ok
        d_embedding<<<grid, block, 0, s1[i][s]>>>(j, d_decoder_input[i]+ s * num_per_stream * MAX_LENGTH
                                                , d_dw_emb[i],
                                                 d_decoder_embedded[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        dim3 block2(TILE_SIZE, MAX_LENGTH);
        dim3 grid2(num_per_stream / TILE_SIZE, 1);
        d_concat_linear<<<grid2, block2, 0, s1[i][s]>>>(d_decoder_embedded[i]+ s * num_per_stream * HIDDEN_SIZE, d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                    , d_dw_attn[i], d_db_attn[i],
                                                     d_decoder_attn_linear[i] + s * num_per_stream * MAX_LENGTH, d_enable[i]+ s * num_per_stream);
        int aa = (N > 8192) ? 16 : 1;
        dim3 block3(TILE_SIZE * aa);
        dim3 grid3(num_per_stream / TILE_SIZE / aa);
        d_softmax<<<grid3, block3, 0, s1[i][s]>>>(d_decoder_attn_linear[i]+ s * num_per_stream * MAX_LENGTH
                                              , d_decoder_attn_weights[i]+ s * num_per_stream * MAX_LENGTH
                                              , d_enable[i]+ s * num_per_stream);
        dim3 block4(TILE_SIZE, TILE_SIZE);
        dim3 grid4(num_per_stream / TILE_SIZE, HIDDEN_SIZE / TILE_SIZE);
        d_bmm<<<grid4, block4, 0, s1[i][s]>>>(d_decoder_attn_weights[i]+ s * num_per_stream * MAX_LENGTH,
                                               d_encoder_outputs[i]+ s * num_per_stream * MAX_LENGTH * HIDDEN_SIZE,
                                                d_decoder_attn_applied[i]+ s * num_per_stream * HIDDEN_SIZE,
                                                 d_enable[i]+ s * num_per_stream);
        d_concat_linear_relu<<<grid4, block4, 0, s1[i][s]>>>(d_decoder_embedded[i]+ s * num_per_stream * HIDDEN_SIZE,
                                                         d_decoder_attn_applied[i]+ s * num_per_stream * HIDDEN_SIZE,
                                                          d_dw_attn_comb[i], d_db_attn_comb[i],
                                                           d_decoder_relu[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        d_wandb_improved<<<grid4, block4, 0, s1[i][s]>>>(d_decoder_relu[i]+ s * num_per_stream * HIDDEN_SIZE, d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                    , d_dw_ir[i], d_dw_hr[i], d_db_ir[i], d_db_hr[i],
                                                     d_decoder_rt[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        d_wandb_improved<<<grid4, block4, 0, s1[i][s]>>>(d_decoder_relu[i]+ s * num_per_stream * HIDDEN_SIZE,
                                                     d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                     , d_dw_iz[i], d_dw_hz[i], d_db_iz[i], d_db_hz[i], 
                                                     d_decoder_zt[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        d_wandb_improved_mul<<<grid4, block4, 0, s1[i][s]>>>(d_decoder_relu[i]+ s * num_per_stream * HIDDEN_SIZE, d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE
                                                      , d_decoder_rt[i]+ s * num_per_stream * HIDDEN_SIZE, d_dw_in[i], d_dw_hn[i], d_db_in[i], d_db_hn[i], 
                                                      d_decoder_nt[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        dim3 block7(TILE_SIZE, TILE_SIZE);
        dim3 grid7(num_per_stream / TILE_SIZE, HIDDEN_SIZE / 4 / TILE_SIZE);
        // ok
        d_elemwise_op<<<grid7, block7, 0, s1[i][s]>>>(d_decoder_zt[i]+ s * num_per_stream * HIDDEN_SIZE, d_decoder_nt[i]+ s * num_per_stream * HIDDEN_SIZE
                                                   , d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE, d_enable[i]+ s * num_per_stream);
        dim3 block5(TILE_SIZE, TILE_SIZE);
        dim3 grid5(num_per_stream / TILE_SIZE, (OUTPUT_VOCAB_SIZE + TILE_SIZE - 1) / TILE_SIZE);
        d_wandb_fin<<<grid5, block5, 0, s1[i][s]>>>(d_encoder_hidden[i]+ s * num_per_stream * HIDDEN_SIZE, d_dw_out[i], d_db_out[i], d_decoder_fin[i]+ s * num_per_stream * OUTPUT_VOCAB_SIZE, d_enable[i]+ s * num_per_stream);
        
        dim3 block6(TILE_SIZE * aa);
        dim3 grid6(num_per_stream / TILE_SIZE / aa);
        // ok
        d_top_one_soft<<<grid6, block6, 0, s1[i][s]>>>(d_decoder_fin[i]+ s * num_per_stream * OUTPUT_VOCAB_SIZE, d_output[i]+ s * num_per_stream * MAX_LENGTH, d_decoder_input[i]+ s * num_per_stream * MAX_LENGTH, j, d_enable[i]+ s * num_per_stream);
      }
      // cudaMemcpyAsync((output->buf) + (i * num_per_gpu + s * num_per_stream) * MAX_LENGTH, d_output[i] + s * num_per_stream * MAX_LENGTH, num_per_stream * MAX_LENGTH * sizeof(float), cudaMemcpyDeviceToHost, s1[i][s]);
    }
  }

  for (int i=0; i<num_devices; i++) {
    cudaSetDevice(i);
    for (int s=0; s<NUM_STREAM; s++) {
      cudaStreamSynchronize(s1[i][s]);
    }
    cudaMemcpy((output->buf) + i * num_per_gpu * MAX_LENGTH, d_output[i], num_per_gpu * MAX_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  }




  MPI_Gather((void *)(output->buf), num_per_node * MAX_LENGTH, MPI_FLOAT, (void *)(output->buf), num_per_node * MAX_LENGTH, MPI_FLOAT , 0, MPI_COMM_WORLD);
}


/*
 * initialize_translator
 * @brief : initialize translator. load the parameter binary file and store parameters into Tensors
 *          
 * @param [in1] parameter_fname  : the name of the binary file where parameters are stored
 */
void initialize_translator(const char *parameter_fname, int N){
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  cudaGetDeviceCount(&num_devices);

  num_per_node = N / mpi_size;
  num_per_gpu = num_per_node / num_devices;
  num_per_stream = num_per_gpu / NUM_STREAM;
  printf("rank [%d] gpu %d\n", mpi_rank, num_devices);
  fflush(stdout);

  // parameter sharing
  if (mpi_rank == 0) {
    size_t parameter_binary_size = 0;
    float *parameter = (float *) read_binary(parameter_fname, &parameter_binary_size);
    eW_emb = new Tensor({INPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET0);
    eW_ir = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET1);
    eW_iz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET2);
    eW_in = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET3);
    eW_hr = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET4);
    eW_hz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET5);
    eW_hn = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET6);
    eb_ir = new Tensor({HIDDEN_SIZE}, parameter + OFFSET7);
    eb_iz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET8);
    eb_in = new Tensor({HIDDEN_SIZE}, parameter + OFFSET9);
    eb_hr = new Tensor({HIDDEN_SIZE}, parameter + OFFSET10);
    eb_hz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET11);
    eb_hn = new Tensor({HIDDEN_SIZE}, parameter + OFFSET12);
    dW_emb = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET13);
    dW_ir = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET14);
    dW_iz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET15);
    dW_in = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET16);
    dW_hr = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET17);
    dW_hz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET18);
    dW_hn = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET19);
    db_ir = new Tensor({HIDDEN_SIZE}, parameter + OFFSET20);
    db_iz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET21);
    db_in = new Tensor({HIDDEN_SIZE}, parameter + OFFSET22);
    db_hr = new Tensor({HIDDEN_SIZE}, parameter + OFFSET23);
    db_hz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET24);
    db_hn = new Tensor({HIDDEN_SIZE}, parameter + OFFSET25);
    dW_attn = new Tensor({MAX_LENGTH, 2 * HIDDEN_SIZE}, parameter + OFFSET26);
    db_attn = new Tensor({MAX_LENGTH}, parameter + OFFSET27);
    dW_attn_comb = new Tensor({HIDDEN_SIZE, 2 * HIDDEN_SIZE}, parameter + OFFSET28);
    db_attn_comb = new Tensor({HIDDEN_SIZE}, parameter + OFFSET29);
    dW_out = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET30);
    db_out = new Tensor({OUTPUT_VOCAB_SIZE}, parameter + OFFSET31);
    usleep(1);
  } else {
    eW_emb = new Tensor({INPUT_VOCAB_SIZE, HIDDEN_SIZE});
    eW_ir = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_iz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_in = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_hr = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_hz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_hn = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eb_ir = new Tensor({HIDDEN_SIZE});
    eb_iz = new Tensor({HIDDEN_SIZE});
    eb_in = new Tensor({HIDDEN_SIZE});
    eb_hr = new Tensor({HIDDEN_SIZE});
    eb_hz = new Tensor({HIDDEN_SIZE});
    eb_hn = new Tensor({HIDDEN_SIZE});
    dW_emb = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE});
    dW_ir = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_iz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_in = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_hr = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_hz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_hn = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    db_ir = new Tensor({HIDDEN_SIZE});
    db_iz = new Tensor({HIDDEN_SIZE});
    db_in = new Tensor({HIDDEN_SIZE});
    db_hr = new Tensor({HIDDEN_SIZE});
    db_hz = new Tensor({HIDDEN_SIZE});
    db_hn = new Tensor({HIDDEN_SIZE});
    dW_attn = new Tensor({MAX_LENGTH, 2 * HIDDEN_SIZE});
    db_attn = new Tensor({MAX_LENGTH});
    dW_attn_comb = new Tensor({HIDDEN_SIZE, 2 * HIDDEN_SIZE});
    db_attn_comb = new Tensor({HIDDEN_SIZE});
    dW_out = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE});
    db_out = new Tensor({OUTPUT_VOCAB_SIZE});
  }

  for (int i=0; i<num_devices; i++) {
    cudaSetDevice(i);
    cudaStreamCreate(&s0[i]);
    for (int s=0; s<NUM_STREAM; s++) {
      cudaStreamCreate(&s1[i][s]);
    }
      // encoder
    cudaMalloc(&d_ew_emb[i], INPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_ew_ir[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_ew_iz[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_ew_in[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_ew_hr[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_ew_hz[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_ew_hn[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_eb_ir[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_eb_iz[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_eb_in[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_eb_hr[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_eb_hz[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_eb_hn[i], HIDDEN_SIZE * sizeof(float));

    cudaMalloc(&d_input[i], num_per_gpu * MAX_LENGTH * sizeof(float));
    cudaMalloc(&d_encoder_embedded[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_encoder_hidden[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_encoder_rt[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_encoder_zt[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_encoder_nt[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_encoder_outputs[i], num_per_gpu * MAX_LENGTH * HIDDEN_SIZE * sizeof(float));

    // decoder
    cudaMalloc(&d_dw_emb[i], OUTPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_ir[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_iz[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_in[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_hr[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_hz[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_hn[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_ir[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_iz[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_in[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_hr[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_hz[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_hn[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_attn[i], MAX_LENGTH * 2 * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_attn[i], MAX_LENGTH * sizeof(float));
    cudaMalloc(&d_dw_attn_comb[i], HIDDEN_SIZE * 2 * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_attn_comb[i], HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dw_out[i], OUTPUT_VOCAB_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db_out[i], OUTPUT_VOCAB_SIZE * sizeof(float));

    cudaMalloc(&d_decoder_input[i], num_per_gpu * MAX_LENGTH * sizeof(float));
    cudaMalloc(&d_decoder_embedded[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_decoder_attn_linear[i], num_per_gpu * MAX_LENGTH * sizeof(float));
    cudaMalloc(&d_decoder_attn_weights[i], num_per_gpu * MAX_LENGTH * sizeof(float));
    cudaMalloc(&d_decoder_attn_applied[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_decoder_relu[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_decoder_rt[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_decoder_zt[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_decoder_nt[i], num_per_gpu * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_decoder_fin[i], num_per_gpu * OUTPUT_VOCAB_SIZE * sizeof(float));
    cudaMalloc(&d_output[i], num_per_gpu * MAX_LENGTH * sizeof(float));
    cudaMalloc(&d_enable[i], num_per_gpu * sizeof(int));
  }
}

/*
 * finalize_translator
 * @brief : free all dynamically allocated variables
 */
void finalize_translator(){


  // free parameters
  delete eW_emb;
  delete eW_ir; 
  delete eW_iz; 
  delete eW_in; 
  delete eW_hr; 
  delete eW_hz; 
  delete eW_hn; 
  delete eb_ir; 
  delete eb_iz; 
  delete eb_in; 
  delete eb_hr; 
  delete eb_hz; 
  delete eb_hn; 
  delete dW_emb;
  delete dW_ir; 
  delete dW_iz; 
  delete dW_in; 
  delete dW_hr; 
  delete dW_hz; 
  delete dW_hn; 
  delete db_ir; 
  delete db_iz; 
  delete db_in; 
  delete db_hr; 
  delete db_hz; 
  delete db_hn; 
  delete dW_attn;
  delete db_attn;
  delete dW_attn_comb;
  delete db_attn_comb;
  delete dW_out;
  delete db_out;

  for (int i=0; i<num_devices; i++) {
    cudaSetDevice(i);
    
    cudaStreamDestroy(s0[i]);
    for (int s=0; s<NUM_STREAM; s++) {
      cudaStreamDestroy(s1[i][s]);
    }

    // encoder
    cudaFree(d_ew_emb[i]);
    cudaFree(d_ew_ir[i]);
    cudaFree(d_ew_iz[i]);
    cudaFree(d_ew_in[i]);
    cudaFree(d_ew_hr[i]);
    cudaFree(d_ew_hz[i]);
    cudaFree(d_ew_hn[i]);
    cudaFree(d_eb_ir[i]);
    cudaFree(d_eb_iz[i]);
    cudaFree(d_eb_in[i]);
    cudaFree(d_eb_hr[i]);
    cudaFree(d_eb_hz[i]);
    cudaFree(d_eb_hn[i]);

    cudaFree(d_input[i]);
    cudaFree(d_encoder_embedded[i]);
    cudaFree(d_encoder_hidden[i]);
    cudaFree(d_encoder_rt[i]);
    cudaFree(d_encoder_zt[i]);
    cudaFree(d_encoder_nt[i]);
    cudaFree(d_encoder_outputs[i]);

    // decoder
    cudaFree(d_dw_emb[i]);
    cudaFree(d_dw_ir[i]);
    cudaFree(d_dw_iz[i]);
    cudaFree(d_dw_in[i]);
    cudaFree(d_dw_hr[i]);
    cudaFree(d_dw_hz[i]);
    cudaFree(d_dw_hn[i]);
    cudaFree(d_db_ir[i]);
    cudaFree(d_db_iz[i]);
    cudaFree(d_db_in[i]);
    cudaFree(d_db_hr[i]);
    cudaFree(d_db_hz[i]);
    cudaFree(d_db_hn[i]);

    cudaFree(d_decoder_input[i]);
    cudaFree(d_decoder_embedded[i]);
    cudaFree(d_decoder_attn_linear[i]);
    cudaFree(d_decoder_attn_weights[i]);
    cudaFree(d_decoder_attn_applied[i]);
    cudaFree(d_decoder_relu[i]);
    cudaFree(d_decoder_rt[i]);
    cudaFree(d_decoder_zt[i]);
    cudaFree(d_decoder_nt[i]);
    cudaFree(d_decoder_fin[i]);
    cudaFree(d_output[i]);
    cudaFree(d_enable[i]);
  }
}
