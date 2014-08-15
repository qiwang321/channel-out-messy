#include <cuda.h>
#include "routines.cuh"


NVMask::NVMask() {
	mask = NULL;
	size = 0;
}
void NVMask::resize(NVMatrix& like) {
	resize(like.getNumElements());
}

void NVMask::resize(int len) {
	if (size != len) {
		if (size > 0) {
			cublasStatus status = cublasFree(mask);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! NVMask memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}
		}
		size = len;
		if (size > 0){
			cublasStatus status = cublasAlloc(size, sizeof(int), (void**) &mask);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! NVMask device memory allocation error\n");
				exit(EXIT_FAILURE);
			}
		}
		else {
			mask = NULL;
		}
	}
}

int* NVMask::getDevData() {
	return mask;
}
int NVMask::getSize() {
	return size;
}

void NVMask::print(int len) {
	int* hmMask = (int*) malloc (len*sizeof(int));
	CUDA_CALL(cudaMemcpy(hmMask, mask, len*sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < len; i++)
		printf("%d ", hmMask[i]);
	printf("\n");
	free(hmMask);
}

int* NVMask::copyToHost(int len) {
	int* hmMask = (int*) malloc (len*sizeof(int));
	CUDA_CALL(cudaMemcpy(hmMask, mask, len*sizeof(int), cudaMemcpyDeviceToHost));
	return hmMask;
}

int* NVMask::copyToHost() {
	return copyToHost(size);
}

NVMask::~NVMask() {
	if (size > 0) {
		cublasStatus status = cublasFree(mask);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! NVMask memory free error: %X\n", status);
			exit(EXIT_FAILURE);
		}
	}
}
