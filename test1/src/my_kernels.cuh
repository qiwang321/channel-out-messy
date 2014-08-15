#include <cuda.h>
#include <stdlib.h>
#include <cfloat>

/*
 * assemble several matrices(of same size) into a large matrix
 */
__global__ void kAssemble(float* dest, float* src, int start, int leadingDim, int followingDim, int n, int axis, bool trans) {
	int srcIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int destIdx;
	if ((axis == 0 && trans == false) || (axis == 1 && trans == true)) {
		for (; srcIdx < leadingDim * followingDim; srcIdx += blockDim.x * gridDim.x) {
			destIdx = srcIdx + start * leadingDim * followingDim;
			dest[destIdx] = src[srcIdx];
		}
	}
	else {
		for (; srcIdx < leadingDim * followingDim; srcIdx += blockDim.x * gridDim.x) {
			destIdx = srcIdx + srcIdx / leadingDim * (n-1) * leadingDim + start * leadingDim;
			dest[destIdx] = src[srcIdx];
		}
	}
}

/*
 * assemble two matrices into a large matrix
 */
__global__ void kAssemble(float* dest, float* src1, float* src2, int l1, int f1, int l2, int f2, int axis, bool trans) {
	int srcIdx;
	int destIdx;
	if ((axis == 0 && trans == false) || (axis == 1 && trans == true)) {
		for (srcIdx = blockIdx.x * blockDim.x + threadIdx.x;
				srcIdx < l1 * f1; srcIdx += blockDim.x * gridDim.x) {
			destIdx = srcIdx;
			dest[destIdx] = src1[srcIdx];
		}
		for (srcIdx = blockIdx.x * blockDim.x + threadIdx.x;
				srcIdx < l2 * f2; srcIdx += blockDim.x * gridDim.x) {
			destIdx = srcIdx + l1 * f1;
			dest[destIdx] = src2[srcIdx];
		}
	}
	else {
		for (srcIdx = blockIdx.x * blockDim.x + threadIdx.x;
				srcIdx < l1 * f1; srcIdx += blockDim.x * gridDim.x) {
			destIdx = srcIdx + srcIdx / l1 * l2;
			dest[destIdx] = src1[srcIdx];
		}
		for (srcIdx = blockIdx.x * blockDim.x + threadIdx.x;
				srcIdx < l2 * f2; srcIdx += blockDim.x * gridDim.x) {
			destIdx = srcIdx + srcIdx / l2 * l1 + l1;
			dest[destIdx] = src2[srcIdx];
		}
	}
}

/*
 * split a large matrix into several small matrices (of same size)
 */
__global__ void kSplit(float* src, float* dest, int start, int leadingDim, int followingDim, int n, int axis, bool trans) {
	int destIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int srcIdx;
	if ((axis == 0 && trans == false) || (axis == 1 && trans == true)) {
		for (; destIdx < leadingDim * followingDim; destIdx += blockDim.x * gridDim.x) {
			srcIdx = destIdx + start * leadingDim * followingDim;
			dest[destIdx] = src[srcIdx];
		}
	}
	else {
		for (; destIdx < leadingDim * followingDim; destIdx += blockDim.x * gridDim.x) {
			srcIdx = destIdx + destIdx / leadingDim * (n-1) * leadingDim + start * leadingDim;
			dest[destIdx] = src[srcIdx];
		}
	}
}

/*
 * separate a large matrix into two matrices
 */
__global__ void kSplit(float* src, float* dest1, float* dest2, int l1, int f1, int l2, int f2, int axis, bool trans) {
	int srcIdx;
	int destIdx;
	if ((axis == 0 && trans == false) || (axis == 1 && trans == true)) {
		for (destIdx = blockIdx.x * blockDim.x + threadIdx.x;
				destIdx < l1 * f1; destIdx += blockDim.x * gridDim.x) {
			srcIdx = destIdx;
			dest1[destIdx] = src[srcIdx];
		}
		for (destIdx = blockIdx.x * blockDim.x + threadIdx.x;
				destIdx < l2 * f2; destIdx += blockDim.x * gridDim.x) {
			srcIdx = destIdx + l1 * f1;
			dest2[destIdx] = src[srcIdx];
		}
	}
	else {
		for (destIdx = blockIdx.x * blockDim.x + threadIdx.x;
				destIdx < l1 * f1; destIdx += blockDim.x * gridDim.x) {
			srcIdx = destIdx + destIdx / l1 * l2;
			dest1[destIdx] = src[srcIdx];
		}
		for (destIdx = blockIdx.x * blockDim.x + threadIdx.x;
				destIdx < l2 * f2; destIdx += blockDim.x * gridDim.x) {
			srcIdx = destIdx + destIdx / l2 * l1 + l1;
			dest2[destIdx] = src[srcIdx];
		}
	}
}

/*
 * initialize cuda rand
 */
__global__ void kRandSetup(curandState *state, int seed) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

/*
 * generate filter-scale feature mask
 */
__global__ void kFilterMask(float* data, int numRows, int numCols, float prob, curandState* states) {
	// blockIdx is col index, threadIdx is row index
	__shared__ float mask;
	if (threadIdx.x == 0) {
		if (prob > curand_uniform(&states[blockIdx.x]))
			mask = 1.0f;
		else
			mask = 0.0f;
	}
	__syncthreads();
	for (int i = threadIdx.x; i < numRows; i+=blockDim.x) {
		data[i*numCols+blockIdx.x] = mask;
	}
}

/*
 * generate random binary matrix according to uniform probability
 */
__global__ void kRandBinMat(float* data, int numRows, int numCols, float prob, curandState* states) {
	// blockIdx is col index, threadIdx is row index
	for (int i = threadIdx.x; i < numRows; i += blockDim.x)
		for (int j = blockIdx.x; j < numCols; j += gridDim.x) {
			if (prob > curand_uniform(&states[threadIdx.x * gridDim.x + blockIdx.x]))
				data[i*numCols+j] = 1.0f;
			else
				data[i*numCols+j] = 0.0f;
		}
}

/*
 * maxout operation kernel
 * numColors must be divisible by poolSize
 * inData(numColors, numPixels, numCases)
 * outData(numColors/poolSize, numPixels, numCases)
 * gridDim.x = numModules blockDim.x = numCases
 */
__global__ void kMaxOut(float* inData, float* outData, int poolSize, int poolStride, int numGroups, int numColors) {

	float max;
	float tmp;
	int idx_in, idx_out;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;

	for (int i = 0; i < numGroups; i++) {
		max = - FLT_MAX;
		idx_in = idx + i * poolStride * dim2;
		for (int j = 0; j < poolSize; j++) {
			tmp = inData[idx_in];
			if (tmp > max)
				max = tmp;
			idx_in += dim2;
			if (idx_in >= numColors * dim2)
				idx_in -= numColors * dim2;
		}
		idx_out = idx + i * dim2;
		outData[idx_out] = max;
	}
}

__global__ void kMaxOut(float* inData, float* outData, int* maskData, int poolSize, int poolStride, int numGroups, int numColors) {

	float max;
	float tmp;
	int max_id;
	int idx_in, idx_out;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;

	for (int i = 0; i < numGroups; i++) {
		max = - FLT_MAX; max_id = -1;
		idx_in = idx + i * poolStride * dim2;
		idx_out = idx + i * dim2;
		for (int j = 0; j < poolSize; j++) {
			tmp = inData[idx_in];
			if (tmp > max) {
				max = tmp;
				max_id = j;
			}
			idx_in += dim2;
			if (idx_in >= numColors * dim2)
				idx_in -= numColors * dim2;
		}
		maskData[idx_out] = max_id;// here
		outData[idx_out] = max;
	}
}

__global__ void kCompeteOut(float* inData, float* outData, int* maskData, int poolSize, int poolStride, int numGroups, int numColors) {

	float max;
	float tmp;
	int max_id;
	int idx_in, idx_max, idx_ind;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;

	// initialize
	idx_in = idx;
	for (int i = 0; i < numColors; i++) {
		outData[idx_in] = 0.0f;
		idx_in += dim2;
	}

	for (int i = 0; i < numGroups; i++) {
		max = - FLT_MAX; max_id = -1;
		idx_in = idx + i * poolStride * dim2;
		idx_ind = idx + i * dim2;
		for (int j = 0; j < poolSize; j++) {
			tmp = inData[idx_in];
			if (tmp > max) {
				max = tmp;
				max_id = j;
				idx_max = idx_in;
			}
			idx_in += dim2;
			if (idx_in >= numColors * dim2)
				idx_in -= numColors * dim2;
		}
		maskData[idx_ind] = max_id;// here
		outData[idx_max] += max;
	}
}

/*
 * maxout gradient
 * numColors must be divisible by poolSize
 * gradData(numColors/poolSize, numPixels, numCases)
 * targetData(numColors, numPixels, numCases)
 * gridDim.x = numModules blockDim.x = numCases
 */
__global__ void kMaxOutUndo(float* gradData, float* targetData, float* imageData, float* maxData, int poolSize, int poolStride, int numGroups, int numColors) {

	float grad, max;
	int idx_grad, idx_target;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;
	// initialize
	idx_target = idx;
	for (int i = 0; i < numColors; i++) {
		targetData[idx_target] = 0.0f;
		idx_target += dim2;
	}
	// real work
	for (int i = 0; i < numGroups; i++) {
		idx_grad = idx + i * dim2;
		grad = gradData[idx_grad];
		max = maxData[idx_grad];
		idx_target = idx + i * poolStride * dim2;
		for (int j = 0; j < poolSize; j++) {
			targetData[idx_target] += (max == imageData[idx_target]) * grad;
			idx_target += dim2;
			if (idx_target >= numColors * dim2)
				idx_target -= numColors * dim2;
		}
	}
}

__global__ void kMaxOutUndo(float* gradData, float* targetData, int* maskData, int poolStride, int numGroups, int numColors) {

	int idx_grad, idx_target;
	int max_id;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;
	// initialize
	idx_target = idx;
	for (int i = 0; i < numColors; i++) {
		targetData[idx_target] = 0.0f;
		idx_target += dim2;
	}
	// real work
	for (int i = 0; i < numGroups; i++) {
		idx_grad = idx + i * dim2;
		max_id = maskData[idx_grad];
		idx_target = idx + (i * poolStride + max_id) * dim2;
		if (idx_target >= numColors * dim2)
			idx_target -= numColors * dim2;
		targetData[idx_target] += gradData[idx_grad];
	}
}

__global__ void kCompeteOutUndo(float* gradData, float* targetData, int* maskData, int poolStride, int numGroups, int numColors) {

	int idx_ind, idx_target;
	int max_id;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;
	// initialize
	idx_target = idx;
	for (int i = 0; i < numColors; i++) {
		targetData[idx_target] = 0.0f;
		idx_target += dim2;
	}
	// real work
	for (int i = 0; i < numGroups; i++) {
		idx_ind = idx + i * dim2;
		max_id = maskData[idx_ind];
		if (max_id >= 0) {
			idx_target = idx + (i * poolStride + max_id) * dim2;
			if (idx_target >= numColors * dim2)
				idx_target -= numColors * dim2;
			targetData[idx_target] += gradData[idx_target];
		}
	}
}

__global__ void kCompeteAbs(float* inData, float* outData, int* maskData, int poolSize, int poolStride, int numGroups, int numColors) {

	float max;
	float tmp;
	int max_id;
	int idx_in, idx_max, idx_ind;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx in the batchsize and pixels dimensions
	const int dim2 = gridDim.x * blockDim.x;

	// initialize
	idx_in = idx;
	for (int i = 0; i < numColors; i++) {
		outData[idx_in] = 0.0f;
		idx_in += dim2;
	}

	for (int i = 0; i < numGroups; i++) {
		max = 0.0; max_id = -1;
		idx_in = idx + i * poolStride * dim2;
		idx_ind = idx + i * dim2;
		for (int j = 0; j < poolSize; j++) {
			tmp = inData[idx_in];
			if (fabsf(tmp) > fabsf(max)) {
				max = tmp;
				max_id = j;
				idx_max = idx_in;
			}
			idx_in += dim2;
			if (idx_in >= numColors * dim2)
				idx_in -= numColors * dim2;
		}
		maskData[idx_ind] = max_id;// here
		outData[idx_max] += max;
	}
}
