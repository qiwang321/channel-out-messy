/*
 * routines.cuh

 *
 *  Created on: Jul 3, 2013
 *      Author: qwang37
 */

#ifndef ROUTINES_CUH_
#define ROUTINES_CUH_

#include <nvmatrix.cuh>
#include <string>
#include "opt.cuh"
#include <cuda.h>
#include <vector>

class LogisticGradientOperator {
public:
    __device__ inline float operator()(float unitActGrad, float unitAct) const {
        return unitActGrad * unitAct * (1.0f - unitAct);
    }
};

class ReluOperator {
public:
    __device__ inline float operator()(float x) const {
        return x < 0.0f ? 0.0f : x;
    }
};

class ReluGradientOperator {
public:
    __device__ inline float operator()(float unitActGrad, float unitAct) const  {
        return unitActGrad * (unitAct > 0.0f);
    	//return unitActGrad * (unitAct > 0.0f || unitActGrad > 0.0f);
    }
};

class NVMask {
private:
	int* mask;
	int size;
public:
	NVMask();
	void resize(int len);
	void resize(NVMatrix& like);
	int* getDevData();
	int getSize();
	void print(int len);
	int* copyToHost(int len);
	int* copyToHost();
	~NVMask();
};


void samplePatches(int patchSize, int numPatches, int dims[], int numRecords, char* in_name, char* out_name);
void hmSaveToFile(Matrix& hm, const char* fileName, bool append);
void hmSaveToFile(Matrix& hm, string fileName, bool append);
void NVSaveToFile(NVMatrix& dm, const char* fileName, bool append);
void NVSaveToFile(NVMatrix& dm, const char* fileName);
void NVSaveToFile(NVMatrix& dm, string fileName, bool append);
void NVSaveToFile(NVMatrix& dm, string fileName);
void hmReadFromFile(Matrix& hm, const char* fileName, int startRecord);
void hmReadFromFile(Matrix& hm, string fileName, int startRecord);
void hmReadFromFileUint8(Matrix& hm, const char* fileName, int startRecord);
void hmReadFromFileUint8(Matrix& hm, string fileName, int startRecord);
void NVReadFromFile(NVMatrix& dm, const char* fileName, int startRecord);
void NVReadFromFile(NVMatrix& dm, const char* fileName);
void NVReadFromFile(NVMatrix& target, string fileName);
void NVReadFromFile(NVMatrix& target, string fileName, int startRecord);
void NVReadFromFileUint8(NVMatrix& target, const char* fileName, int startRecord);
void NVReadFromFileUint8(NVMatrix& target, const char* fileName);
void hmLabelReadFromFile(Matrix& hm, const char* fileName, int startRecord);
void NVLabelReadFromFile(NVMatrix& dm, const char* fileName, int startRecord);
void NVLabelReadFromFile(NVMatrix& dm, const char* fileName);
void hmRawLabelReadFromFile(Matrix& target, const char* fileName, int startRecord);
void NVRawLabelReadFromFile(NVMatrix& target, const char* fileName, int startRecord);
void NVRawLabelReadFromFile(NVMatrix& target, const char* fileName);
void NVRawLabelReadFromFile(NVMatrix& target, string fileName, int startRecord);
void NVRawLabelReadFromFile(NVMatrix& target, string fileName);

void initWeights(NVMatrix& weights, int numRows, int numCols, bool trans, MTYPE stv);
void updateWeight(NVMatrix& weightGrad, NVMatrix& weightInc, NVMatrix& weight, LayerOpt& opt, int batchSize, float lr_scale, float mom_scale);
void updateBias(NVMatrix& biasGrad, NVMatrix& biasInc, NVMatrix& bias, LayerOpt& opt, int batchSize, float lr_scale, float mom_scale);
float lrDecay(float rate, char* type, float factor, float minRate);
float momInc(float rate, char* type, float factor, float maxRate);

//void trainFCAE(LayerOpt& opt, char* weightFile, char* biasFile);
//void generateData(char* sourceFile, char* destFile, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt);

void activate(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, MTYPE scaleTarget, MTYPE scaleAB);
void activateConv(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt);
void activateConvNoShare(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt);
void activateDual(NVMatrix& src, NVMatrix& destP, NVMatrix& destN, NVMatrix& weight, NVMatrix& biasP, NVMatrix biasN, MTYPE scaleTarget, MTYPE scaleAB);
void activateConvDual(NVMatrix& src, NVMatrix& destP, NVMatrix& destN, NVMatrix& weight, NVMatrix& biasP, NVMatrix& biasN, LayerOpt& opt);
void activateLocal(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt);
void cropDataProvider(vector<Matrix*>& CPUData, vector<NVMatrix*>& GPUData, LayerOpt& opt, bool test, bool whitened);
void multiViewDataProvider(vector<Matrix*>& CPUData, vector<NVMatrix*>& GPUData, LayerOpt& opt, int numViews, bool whitened);

void assembleNVMatrix(vector<NVMatrix>& matrices, NVMatrix& target, int axis);
void assembleNVMatrix(NVMatrix& mat1, NVMatrix& mat2, NVMatrix& target, int axis);
void splitNVMatrix(vector<NVMatrix>& targets, NVMatrix& mat, int axis);
void splitNVMatrix(NVMatrix& t1, NVMatrix& t2, NVMatrix& mat, int n1, int n2, int axis);

// cuda utils
void genFilterMask(NVMatrix& target, int numRows, int numCols, MTYPE prob, curandState* devStates);
void genRandBinMatrix(NVMatrix& target, int numRows, int numCols, MTYPE prob, curandState* devStates);
void genRandBinMatrix(NVMatrix& target, NVMatrix& like, MTYPE prob, curandState* devStates);
curandState* init_cuda_rand(int len);
void convMaxOut(NVMatrix& image, NVMatrix& target, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases);
void convMaxOutUndo(NVMatrix& maxGrad, NVMatrix& target, NVMatrix& image, NVMatrix& maxOut, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases);
void convMaxOut(NVMatrix& image, NVMatrix& target, NVMask& mask, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases);
void convMaxOutUndo(NVMatrix& maxGrad, NVMatrix& target, NVMask& mask, int numColors, int poolStride, int imgSizeX, int numCases);
void convCompeteOut(NVMatrix& image, NVMatrix& target, NVMask& mask, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases);
void convCompeteOutUndo(NVMatrix& maxGrad, NVMatrix& target, NVMask& mask, int numColors, int poolStride, int imgSizeX, int numCases);
void convCompeteAbs(NVMatrix& image, NVMatrix& target, NVMask& mask, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases);

void NVNormalizeCol(NVMatrix& mat, float norm_max);
void NVNormalizeCol1(NVMatrix& mat, NVMatrix& normCol, NVMatrix& tmp, NVMatrix& div, float max_norm);
void NVNormalizeCol2(NVMatrix& mat, NVMatrix& bias, NVMatrix& normCol, NVMatrix& tmp, float max_norm);
void scaleWeight(string fileName, float scale, int numRows, int numCols);
void scaleWeights5(string dirName, float scale);





#endif


