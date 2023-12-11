# PDPCode

This repo contains all using code for Parallel And Distributed Programming 2023 project.

## Matrix Multiplication with MPI and SUMMA Algorithm

It's code which does a simple matrix multiplication A x B = C 

You can make with Makefile and test original and SUMMA version with run.sh in matmul directory.

## Gru Translator Inference with CPU and GPU

It is code which does an translating inference from French to English with trained simple Gru Translator Parameter.

You can make with Makefile and test CPU and GPU version with run.sh in translator/cpu, translator/gpu directory.

If you want to use validation mode you should add answer file in translator/cpu/data and translator/gpu/data directory.

answer file can be downloaded from [Google Drive](https://drive.google.com/file/d/1x4_7Nif5t-aPyjliWEFsUAQXG8Vn0pSS/view?usp=sharing
)

## References

https://github.com/zhongyr/SUMMA_MPI

https://github.com/Cjkkkk/CUDA_gemm

https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
