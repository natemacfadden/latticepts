#!/bin/bash

for b in 1 2 3 4 5 6 7 8 9 10; do
    echo "STUDYING" $b
    ./test_conevecs_kernel $b
    echo "DONE STUDYING" $b
done
