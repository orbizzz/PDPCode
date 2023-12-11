CXX=mpic++
CUX=/usr/local/cuda/bin/nvcc

CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mavx512f -fopenmp -I/usr/local/cuda/include -I/usr/local/include -I/usr/include/x86_64-linux-gnu/openmpi
CUDA_CFLAGS:=$(foreach option, $(CFLAGS),-Xcompiler=$(option))
LDFLAGS=-pthread -L/usr/local/cuda/lib64 -L/usr/local/lib
LDLIBS=-lmpi_cxx -lmpi -lstdc++ -lcudart -lm

TARGET=translator
OBJECTS=main.o translator.o util.o 

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CUX) $(CUDA_CFLAGS) -c -o $@ -x cu $^

# %.o: %.cpp
# 	$(CXX) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
