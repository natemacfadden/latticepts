CFLAGS = -O3 -march=native -flto -mtune=native -funroll-loops
IFLAGS = -I../../conevecs

.PHONY: all clean

all: test_box_enum

test_box_enum: test_box_enum.c
	clang $(CFLAGS) $(IFLAGS) $< -o $@

clean:
	rm -f test_box_enum
