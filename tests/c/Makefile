CC ?= cc

# native tuning: arm64 (Apple Silicon) uses -mcpu=native; x86 uses -march=native
ARCH := $(shell uname -m)
ifneq (,$(filter arm64 aarch64,$(ARCH)))
  ARCHFLAGS := -mcpu=native
else
  ARCHFLAGS := -march=native -mtune=native
endif

CFLAGS = -O3 $(ARCHFLAGS) -flto -funroll-loops
IFLAGS = -I../../latticepts

.PHONY: all clean

all: test_box_enum

test_box_enum: test_box_enum.c
	$(CC) $(CFLAGS) $(IFLAGS) $< -o $@

clean:
	rm -f test_box_enum
