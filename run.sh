#!/bin/bash
#PBS -q gpu -l walltime=10:0:0

echo starting
cd ~/nvmatrix_test/test1/Release
./test1
echo ending
