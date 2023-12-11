TARGET=main
OBJECTS=util.o matmul.o
SUMMA_TARGET=main_summa
SUMMA_OBJECTS=util.o matmul_summa.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -fopenmp -mavx512f
LDLIBS=-lm -lpthread -lmpi -lmpi_cxx

CC=mpicc

all: $(TARGET) $(SUMMA_TARGET)

$(TARGET): $(OBJECTS)

$(SUMMA_TARGET): $(SUMMA_OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
