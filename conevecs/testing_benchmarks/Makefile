CFLAGS = -O3 -march=native -flto -mtune=native -funroll-loops
IFLAGS = -I.. -I$(CONDA_PREFIX)/include
LFLAGS = -L$(CONDA_PREFIX)/lib -lgmp -Wl,-rpath,$(CONDA_PREFIX)/lib

all: test_box_enum

test_box_enum: test_box_enum.c
	clang $(CFLAGS) $(IFLAGS) $< -o $@

clean:
	rm -f test_box_enum
