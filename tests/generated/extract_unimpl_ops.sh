#!/bin/bash

python main.py --limit 500 -j 8 | grep "NotImplementedError: Unimplemented torch op in the IREE compiler" | grep -o "'[^']*'" | sed "s/'//g" > unimplemented_torch_ops.txt