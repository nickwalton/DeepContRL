#!/bin/bash
source activate base
export PYTHONUNBUFFERED=yes
python android_exp1.py 0 > exp0.txt &
python android_exp1.py 1 > exp1.txt &
python android_exp1.py 2 > exp2.txt &
python android_exp1.py 3 > exp3.txt &
python android_exp1.py 4 > exp4.txt &
python android_exp1.py 5 > exp5.txt &
python android_exp1.py 6 > exp6.txt &
python android_exp1.py 7 > exp7.txt &
python android_exp1.py 8 > exp8.txt &
python android_exp1.py 9 > exp9.txt &
python android_exp1.py 10 > exp10.txt &
python android_exp1.py 11 > exp11.txt &

