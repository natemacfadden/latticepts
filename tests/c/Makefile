CC ?= cc

# native tuning: arm64 (Apple Silicon) uses -mcpu=native; x86 uses -march=native
ARCH := $(shell uname -m)
ifneq (,$(filter arm64 aarch64,$(ARCH)))
  ARCHFLAGS := -mcpu=native
else
  ARCHFLAGS := -march=native -mtune=native
endif

# -Wall -Wextra gate the hand-written kernel (box_enum.h compiled standalone here,
# so no Cython-generated-C noise). EXTRA_CFLAGS lets CI add -Werror.
CFLAGS = -O3 $(ARCHFLAGS) -flto -funroll-loops -Wall -Wextra $(EXTRA_CFLAGS)
IFLAGS = -I../../latticepts
LDLIBS = -lm   # libm: macOS links it implicitly, Linux does not

.PHONY: all test clean

all: test_box_enum

test_box_enum: test_box_enum.c
	$(CC) $(CFLAGS) $(IFLAGS) $< -o $@ $(LDLIBS)

# build and run the kernel smoke test (nonzero exit on kernel error)
test: test_box_enum
	./test_box_enum 3

clean:
	rm -f test_box_enum
