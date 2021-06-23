
ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -lm

LIBS = -lOpenCL -fopenmp

SRC_DIR = /src

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

CCFLAGS += -D DEVICE=$(DEVICE)

vadd_chain: vadd_chain.c $(SRC_DIR)/device_info.c libcachelib.a
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(SRC_DIR) -o $@

cache.o: cache.c cache.h
	$(CC) -Wno-implicit-function-declaration -I $(SRC_DIR) -O -c cache.c

libcachelib.a: cache.o
	ar rcs libcachelib.a cache.o

clean:
	rm -f vadd_chain *.a *.o
