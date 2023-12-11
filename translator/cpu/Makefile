CXX=mpic++
CUX=/usr/local/cuda/bin/nvcc

CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mavx512f -fopenmp -I/usr/local/cuda/include -I/usr/local/include
CUDA_CFLAGS:=$(foreach option, $(CFLAGS),-Xcompiler=$(option))
LDFLAGS=-pthread -L/usr/local/cuda/lib64 -L/usr/local/lib
LDLIBS=-lmpi_cxx -lmpi -lstdc++ -lcudart -lm

TARGET_ORIGIN=translator_origin
OBJECTS_ORIGIN=main.o translator_origin.o util.o
TARGET=translator
OBJECTS=main.o translator.o util.o
TARGET_MPI=translator_mpi
OBJECTS_MPI=main.o translator_mpi.o util.o


all: $(TARGET_ORIGIN) $(TARGET) $(TARGET_MPI)

$(TARGET_ORIGIN): $(OBJECTS_ORIGIN)
	$(CC) $(CFLAGS) -o $(TARGET_ORIGIN) $(OBJECTS_ORIGIN) $(LDFLAGS) $(LDLIBS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

$(TARGET_MPI): $(OBJECTS_MPI)
	$(CC) $(CFLAGS) -o $(TARGET_MPI) $(OBJECTS_MPI) $(LDFLAGS) $(LDLIBS)

# %.o: %.cpp
# 	$(CUX) $(CUDA_CFLAGS) -c -o $@ -x cu $^

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET_ORIGIN) $(OBJECTS_ORIGIN) $(TARGET) $(OBJECTS)
