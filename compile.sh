export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/nvmatrix_test/lib
nvcc  -I./include -I./include/common -I./include/cudaconv2 -I./include/nvmatrix -L./lib -lcblas -lcublas ./src/*.cu ./src/common/*.cu ./src/cudaconv2/*.cu ./src/nvmatrix/*.cu
