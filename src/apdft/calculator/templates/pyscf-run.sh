#!/bin/bash
export OMP_NUM_THREADS=1
python3 -u run.inp > run.log
rm tmp* &> /dev/null
