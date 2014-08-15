/*
 * train_filter.cu
 *
 *  Created on: Jul 3, 2013
 *      Author: qwang37
 */

#include <iostream>
#include <fstream>
#include <string>
//#include <random>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <assert.h>

#include <nvmatrix.cuh>
#include <cudaconv2.cuh>
#include <conv_util.cuh>
#include "opt.cuh"
#include "routines.cuh"
#include "my_kernels.cuh"
using namespace std;

void samplePatches(int patchSize, int numPatches, int dims[], int numRecords, char* in_name, char* out_name) {
	int patchesPerRecord = numPatches / numRecords;
	ifstream in;
	in.open(in_name, std::ifstream::in | std::ifstream::binary);
	if (in.fail()) {
		printf("data file open failed!\n");
		exit(-1);
	}
	remove(out_name);
	ofstream out;
	out.open(out_name, std::ofstream::out | std::ofstream::binary);
	if (out.fail()) {
		printf("creating output file failed!\n");
		exit(-1);
	}
	int dimall = dims[0]*dims[1]*dims[2];
	int dim2 = dims[0]*dims[1];
	MTYPE* data = (MTYPE*) malloc(dimall*sizeof(MTYPE));

	for (int i = 0; i < numRecords; i++) {
		in.read((char*)data, dimall*sizeof(MTYPE));
		for (int j = 0; j < patchesPerRecord; j++) {
			// data is row-major: pixels->channels->images
			int pixelX = rand() % (dims[0] - patchSize + 1);
			int pixelY = rand() % (dims[1] - patchSize + 1);
			for (int c = 0; c < dims[2]; c++)
				for (int y = 0; y < patchSize; y++)
					for (int x = 0; x < patchSize; x++)
						out.write((char*)(data + c*dim2 + (pixelY+y)*dims[0] + pixelX + x),
								sizeof(MTYPE));
		}
	}
	in.close();
	out.close();
}

// Matrix IO utils
void hmSaveToFile(Matrix& hm, const char* fileName, bool append) {
	ofstream out;
	if (append)
		out.open(fileName, std::ofstream::out | std::ofstream::binary | std::ofstream::app);
	else
		out.open(fileName, std::ofstream::out | std::ofstream::binary);
	if (out.fail()) {
		cout << "open file failed! filename:" << fileName << endl;
		exit(-1);
	}
	// the file format is different from data layout in the matrix!
	int numRecords = hm.getLeadingDim();
	int numDim = hm.getFollowingDim();
	MTYPE* data = hm.getData();
	for (int i = 0; i < numRecords; i++) {
		for (int j = 0; j < numDim; j++) {
			out.write((char*)(data + j*numRecords + i), sizeof(MTYPE));
		}
	}
	out.close();
}

void hmSaveToFile(Matrix& hm, string fileName, bool append) {
	hmSaveToFile(hm, fileName.c_str(), append);
}

void NVSaveToFile(NVMatrix& dm, const char* fileName, bool append) { // dimensions of matrix must be pre-set!
	Matrix hm;
	dm.copyToHost(hm, true); // resize the target matrix before copying
	hmSaveToFile(hm, fileName, append);
}

void NVSaveToFile(NVMatrix& dm, const char* fileName) { // dimensions of matrix must be pre-set!
	NVSaveToFile(dm, fileName, false);
}

void NVSaveToFile(NVMatrix& dm, string fileName, bool append) {
	NVSaveToFile(dm, fileName.c_str(), append);
}
void NVSaveToFile(NVMatrix& dm, string fileName) { // dimensions of matrix must be pre-set!
	NVSaveToFile(dm, fileName, false);
}

void hmReadFromFile(Matrix& target, const char* fileName, int startRecord) {
	ifstream in;
	in.open(fileName, std::ifstream::in | std::ifstream::binary);
	if (in.fail()) {
		cout << "open file failed! filename:" << fileName << endl;
		exit(-1);
	}
	int numRecords = target.getLeadingDim();
	int numDim = target.getFollowingDim();
	MTYPE* data = target.getData();
	in.seekg(startRecord*numDim*sizeof(MTYPE), ios_base::cur); // get to the starting position
	for (int i = 0; i < numRecords; i++) {
		for (int j = 0; j < numDim; j++) {
			in.read((char*)(data + j*numRecords + i), sizeof(MTYPE));
		}
	}
	in.close();
}

void hmReadFromFile(Matrix& target, string fileName, int startRecord) {
	hmReadFromFile(target, fileName.c_str(), startRecord);
}

void hmReadFromFileUint8(Matrix& target, const char* fileName, int startRecord) {
	ifstream in;
	in.open(fileName, std::ifstream::in | std::ifstream::binary);
	if (in.fail()) {
		cout << "open file failed! filename:" << fileName << endl;
		exit(-1);
	}
	int numRecords = target.getLeadingDim();
	int numDim = target.getFollowingDim();
	MTYPE* data = target.getData();
	in.seekg(startRecord*numDim, ios_base::cur); // get to the starting position
	for (int i = 0; i < numRecords; i++) {
		for (int j = 0; j < numDim; j++) {
			*(data + j*numRecords + i) = (MTYPE) in.get();
		}
	}
	in.close();
}

void hmReadFromFileUint8(Matrix& target, string fileName, int startRecord) {
	hmReadFromFileUint8(target, fileName.c_str(), startRecord);
}


void NVReadFromFile(NVMatrix& target, const char* fileName, int startRecord) {
	Matrix hm(target.getNumRows(), target.getNumCols());
	hm.setTrans(target.isTrans());
	hmReadFromFile(hm, fileName, startRecord);
	target.copyFromHost(hm, true);
}

void NVReadFromFile(NVMatrix& target, const char* fileName) {
	NVReadFromFile(target, fileName, 0);
}

void NVReadFromFile(NVMatrix& target, string fileName, int startRecord) {
	NVReadFromFile(target, fileName.c_str(), startRecord);
}

void NVReadFromFile(NVMatrix& target, string fileName) {
	NVReadFromFile(target, fileName, 0);
}

void NVReadFromFileUint8(NVMatrix& target, const char* fileName, int startRecord) {
	Matrix hm(target.getNumRows(), target.getNumCols());
	hm.setTrans(target.isTrans());
	hmReadFromFileUint8(hm, fileName, startRecord);
	target.copyFromHost(hm, true);
}

void NVReadFromFileUint8(NVMatrix& target, const char* fileName) {
	NVReadFromFileUint8(target, fileName, 0);
}

// label reading utility
void hmLabelReadFromFile(Matrix& target, const char* fileName, int startRecord) {
	ifstream in;
	in.open(fileName, std::ifstream::in | std::ifstream::binary);
	if (in.fail()) {
		cout << "open file failed! filename:" << fileName << endl;
		exit(-1);
	}
	int numRecords = target.getLeadingDim();
	int labelSize = target.getFollowingDim();
	MTYPE* data = target.getData();
	char label;
	in.seekg(startRecord, ios_base::beg); // get to the starting position
	for (int i = 0; i < numRecords; i++) {
		label = in.get();
		for (int j = 0; j < labelSize; j++) // right now the number of classes is fixed
			data[j*numRecords + i] = 0.0;
		data[label*numRecords + i] = 1.0;
	}
	in.close();
}

void NVLabelReadFromFile(NVMatrix& target, const char* fileName, int startRecord) {
	Matrix hm(target.getNumRows(), target.getNumCols());
	hm.setTrans(target.isTrans());
	hmLabelReadFromFile(hm, fileName, startRecord);
	target.copyFromHost(hm, true);
}

void NVLabelReadFromFile(NVMatrix& target, const char* fileName)  {
	NVLabelReadFromFile(target, fileName, 0);
}

// read files for raw labes (uint8 class values
void hmRawLabelReadFromFile(Matrix& target, const char* fileName, int startRecord) {
	ifstream in;
	in.open(fileName, std::ifstream::in | std::ifstream::binary);
	if (in.fail()) {
		cout << "open file failed! filename:" << fileName << endl;
		exit(-1);
	}
	int numRecords = target.getLeadingDim();
	MTYPE* data = target.getData();
	in.seekg(startRecord, ios_base::beg); // get to the starting position
	for (int i = 0; i < numRecords; i++) {
		data[i] = MTYPE(in.get());
	}
	in.close();
}

void NVRawLabelReadFromFile(NVMatrix& target, const char* fileName, int startRecord) {
	Matrix hm(target.getNumRows(), target.getNumCols());
	hm.setTrans(target.isTrans());
	hmRawLabelReadFromFile(hm, fileName, startRecord);
	target.copyFromHost(hm, true);
}


void NVRawLabelReadFromFile(NVMatrix& target, const char* fileName)  {
	NVRawLabelReadFromFile(target, fileName, 0);
}

void NVRawLabelReadFromFile(NVMatrix& target, string fileName, int startRecord) {
	NVRawLabelReadFromFile(target, fileName.c_str(), startRecord);
}

void NVRawLabelReadFromFile(NVMatrix& target, string fileName) {
	NVRawLabelReadFromFile(target, fileName, 0);
}

// training utils
MTYPE gaussianRand(MTYPE mean, MTYPE stv) {
	MTYPE u = (MTYPE)rand()/RAND_MAX;
	MTYPE v = (MTYPE)rand()/RAND_MAX;
	MTYPE x = sqrt(-2*log(u)) * cos(2*M_PI*v); // x is gaussian distributed now
	return stv * x + mean;
}

void initWeights(NVMatrix& weights, int numRows, int numCols, bool trans, MTYPE stv) {
	MTYPE* data = (MTYPE*) malloc(numRows*numCols*sizeof(MTYPE));
	for (int i = 0; i < numRows*numCols; i++) {
		data[i] = gaussianRand(0, stv);
	}
	Matrix weightsCPU(data, numRows, numCols, trans);
	weights.copyFromHost(weightsCPU, true);
}

void activate(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, MTYPE scaleTarget, MTYPE scaleAB) {
	src.transpose(true); // make sure that input is column major
	dest.resize(src.getNumRows(), weight.getNumCols());
	dest.setTrans(true);
	dest.addProduct(src, weight, scaleTarget, scaleAB);
	dest.addVector(bias, scaleAB);
}

void activateDual(NVMatrix& src, NVMatrix& destP, NVMatrix& destN, NVMatrix& weight, NVMatrix& biasP, NVMatrix biasN, MTYPE scaleTarget, MTYPE scaleAB) {
	src.transpose(true); // make sure that input is column major
	assert(destP.isTrans()); // dest must be of column type
	assert(destN.isTrans());
	destP.addProduct(src, weight, scaleTarget, scaleAB);
	destP.addVector(biasN, scaleAB, destN);
	destP.addVector(biasP);
	destN.scale(-1.0f);
}

void activateConv(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt) {
	src.transpose(false); // make sure that input is row-major
	dest.transpose(false);
	convFilterActs(src, weight, dest, opt.imSize, opt.outX, opt.outX, opt.paddingStart, 1, opt.numChannels, 1);

	int numFilters = weight.getNumCols();
	int batchSize = src.getNumCols();
	dest.reshape(numFilters, opt.outX * opt.outX * batchSize);
	dest.addVector(bias);
	dest.reshape(numFilters * opt.outX * opt.outX, batchSize);
}

void activateConvNoShare(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt) {
	src.transpose(false); // make sure that input is row-major
	dest.transpose(false);
	convFilterActs(src, weight, dest, opt.imSize, opt.outX, opt.outX, opt.paddingStart, 1, opt.numChannels, 1);

	int numFilters = weight.getNumCols();
	int batchSize = src.getNumCols();
	dest.addVector(bias);
}

void activateConvDual(NVMatrix& src, NVMatrix& destP, NVMatrix& destN, NVMatrix& weight, NVMatrix& biasP, NVMatrix& biasN, LayerOpt& opt) {
	src.transpose(false); // make sure that input is row-major
	destP.transpose(false);
	convFilterActs(src, weight, destP, opt.imSize, opt.outX, opt.outX, opt.paddingStart, 1, opt.numChannels, 1);

	int numFilters = weight.getNumCols();
	int batchSize = src.getNumCols();
	destP.reshape(numFilters, opt.outX * opt.outX * batchSize);
	destP.addVector(biasN, destN);
	destP.addVector(biasP);
	destP.reshape(numFilters * opt.outX * opt.outX, batchSize);
	destN.reshape(numFilters * opt.outX * opt.outX, batchSize);
	destN.scale(-1.0f);
}


void activateLocal(NVMatrix& src, NVMatrix& dest, NVMatrix& weight, NVMatrix& bias, LayerOpt& opt) {
	src.transpose(false); // make sure that input is row-major
	dest.transpose(false);
	localFilterActs(src, weight, dest, opt.imSize, opt.outX, opt.outX, opt.paddingStart, 1, opt.numChannels, 1);
	dest.addVector(bias);
}


void gradSparse(NVMatrix& act, MTYPE desire, NVMatrix& target) {
	act.sum(0, target);
	target.scale(1.0/act.getNumCols());
	target.addScalar(-desire);
}

MTYPE computeSquareCost(NVMatrix& recon, NVMatrix& data, NVMatrix& reconGrad) { // sum-square cost
	recon.subtract(data, reconGrad);
	return reconGrad.norm2();
}

void gradProp(NVMatrix& upperGrad, NVMatrix& targetGrad, NVMatrix& weight) {
	NVMatrix weightT;
	weight.transpose(weightT);
	upperGrad.rightMult(weightT, targetGrad);

}


void computeGrad(NVMatrix& upperGrad, NVMatrix& input, NVMatrix& weightGrad, NVMatrix& biasGrad) {
	NVMatrix inputT;
	input.transpose(inputT);
	inputT.rightMult(upperGrad, weightGrad);
	upperGrad.sum(0, biasGrad);

}


void updateWeight(NVMatrix& weightGrad, NVMatrix& weightInc, NVMatrix& weight, LayerOpt& opt, int batchSize, float lr_scale, float mom_scale) {
	float lr = opt.lrW * lr_scale;
	float mom = opt.mom * mom_scale;

	weightInc.add(weightGrad, mom, lr / batchSize);
	weight.add(weightInc);
}


void updateBias(NVMatrix& biasGrad, NVMatrix& biasInc, NVMatrix& bias, LayerOpt& opt, int batchSize, float lr_scale, float mom_scale) {
	float lr = opt.lrB * lr_scale;
	float mom = opt.mom * mom_scale;

	biasInc.add(biasGrad, mom, lr / batchSize);
	bias.add(biasInc);
}

float lrDecay(float rate, char* type, float factor, float minRate) {
	if (strcmp(type, "linear") == 0) {
		rate = rate - factor;
		return rate > minRate ? rate : minRate;
	}
	if (strcmp(type, "exponential") == 0) {
		rate= rate * factor;
		return rate > minRate ? rate : minRate;
	}
	return 1.0;
}

float momInc(float rate, char* type, float factor, float maxRate) {
	if (strcmp(type, "linear") == 0) {
		rate = rate + factor;
		return rate > maxRate ? maxRate : rate;
	}
	if (strcmp(type, "exponential") == 0) {
		rate= rate * factor;
		return rate > maxRate ? maxRate : rate;
	}
	return 1.0;
}


void cropDataProvider(vector<Matrix*>& CPUData, vector<NVMatrix*>& GPUData, LayerOpt& opt, bool test, bool whitened) {
	if (!whitened) {
		Matrix tmp;
		tmp.setTrans(false);
		int destIdx, srcIdx, meanIdx;
		ifstream in_mean;
		in_mean.open((opt.dataPath + "/data_mean.bin").c_str(), std::ifstream::in | std::ifstream::binary);
		if (in_mean.fail()) {
			cout << "open file failed! filename: " << (opt.dataPath + "/data_mean.bin").c_str() << endl;
			return;
		}
		MTYPE* meanData = (MTYPE*) malloc (3072*sizeof(MTYPE));
		for (int j = 0; j < 3072; j++)
			in_mean.read((char*)(meanData+j), sizeof(MTYPE));
		in_mean.close();
		if (!test) {
			for (int batch = 0; batch < CPUData.size(); batch++) {
				int batchSize = CPUData[batch]->getNumCols();
				tmp.resize(opt.imSize*opt.imSize*opt.numChannels, batchSize);
				MTYPE* destData = tmp.getData();
				MTYPE* srcData = CPUData[batch]->getData();
				for (int l = 0; l < batchSize; l++) {
					int startX = rand() % (32 - opt.imSize + 1);
					int startY = rand() % (32 - opt.imSize + 1);
					int meanStartX = (32 - opt.imSize) / 2;
					int meanStartY = (32 - opt.imSize) / 2;
					int flip;
					if (opt.flip)
						flip = rand() % 2 ;
					else
						flip = 0;

					for (int i = 0; i < opt.imSize; i++)
						for (int j = 0; j < opt.imSize; j++)
							for (int k = 0; k < opt.numChannels; k++) {
								destIdx = ((k*opt.imSize + j) * opt.imSize + i) * batchSize + l;
								if (flip == 0)
									srcIdx = ((k*32 + j + startY) * 32 + i + startX) * batchSize + l;
								else
									srcIdx = ((k*32 + j + startY) * 32 + (opt.imSize - 1 - i) + startX) * batchSize + l;
								meanIdx = (k*32 + j + meanStartY) * 32 + i + meanStartX;
								destData[destIdx] = srcData[srcIdx] - meanData[meanIdx];
							}
				}
				GPUData[batch]->copyFromHost(tmp, true);
			}
		}
		else {
			for (int batch = 0; batch < CPUData.size(); batch++) {
				int batchSize = CPUData[batch]->getNumCols();
				tmp.resize(opt.imSize*opt.imSize*opt.numChannels, batchSize);
				MTYPE* destData = tmp.getData();
				MTYPE* srcData = CPUData[batch]->getData();
				for (int l = 0; l < batchSize; l++) {
					int startX = (32 - opt.imSize) / 2;
					int startY = (32 - opt.imSize) / 2;
					int meanStartX = (32 - opt.imSize) / 2;
					int meanStartY = (32 - opt.imSize) / 2;
					for (int i = 0; i < opt.imSize; i++)
						for (int j = 0; j < opt.imSize; j++)
							for (int k = 0; k < opt.numChannels; k++) {
								destIdx = ((k*opt.imSize + j) * opt.imSize + i) * batchSize + l;
								srcIdx = ((k*32 + j + startY) * 32 + i + startX) * batchSize + l;
								meanIdx = (k*32 + j + meanStartY) * 32 + i + meanStartX;
								destData[destIdx] = srcData[srcIdx] - meanData[meanIdx];
							}
				}
				GPUData[batch]->copyFromHost(tmp, true);
			}
		}
	}
	else {
		Matrix tmp;
		tmp.setTrans(false);
		int destIdx, srcIdx;
		if (!test) {
			for (int batch = 0; batch < CPUData.size(); batch++) {
				int batchSize = CPUData[batch]->getNumCols();
				tmp.resize(opt.imSize*opt.imSize*opt.numChannels, batchSize);
				MTYPE* destData = tmp.getData();
				MTYPE* srcData = CPUData[batch]->getData();
				for (int l = 0; l < batchSize; l++) {
					int startX = rand() % (32 - opt.imSize + 1);
					int startY = rand() % (32 - opt.imSize + 1);
					int flip;
					if (opt.flip)
						flip = rand() % 2;
					else
						flip = 0;

					for (int i = 0; i < opt.imSize; i++)
						for (int j = 0; j < opt.imSize; j++)
							for (int k = 0; k < opt.numChannels; k++) {
								destIdx = ((k*opt.imSize + j) * opt.imSize + i) * batchSize + l;
								if (flip == 0)
									srcIdx = ((k*32 + j + startY) * 32 + i + startX) * batchSize + l;
								else
									srcIdx = ((k*32 + j + startY) * 32 + (opt.imSize - 1 - i) + startX) * batchSize + l;
								destData[destIdx] = srcData[srcIdx];
							}
				}
				GPUData[batch]->copyFromHost(tmp, true);
			}
		}
		else {
			for (int batch = 0; batch < CPUData.size(); batch++) {
				int batchSize = CPUData[batch]->getNumCols();
				tmp.resize(opt.imSize*opt.imSize*opt.numChannels, batchSize);
				MTYPE* destData = tmp.getData();
				MTYPE* srcData = CPUData[batch]->getData();
				for (int l = 0; l < batchSize; l++) {
					int startX = (32 - opt.imSize) / 2;
					int startY = (32 - opt.imSize) / 2;
					for (int i = 0; i < opt.imSize; i++)
						for (int j = 0; j < opt.imSize; j++)
							for (int k = 0; k < opt.numChannels; k++) {
								destIdx = ((k*opt.imSize + j) * opt.imSize + i) * batchSize + l;
								srcIdx = ((k*32 + j + startY) * 32 + i + startX) * batchSize + l;
								destData[destIdx] = srcData[srcIdx];
							}
				}
				GPUData[batch]->copyFromHost(tmp, true);
			}
		}
	}
}

void multiViewDataProvider(vector<Matrix*>& CPUData, vector<NVMatrix*>& GPUData, LayerOpt& opt, int numViews, bool whitened) {
	if (!whitened) {
		Matrix tmp;
		int destIdx, srcIdx, meanIdx;
		ifstream in_mean;
		in_mean.open((opt.dataPath + "/data_mean.bin").c_str(), std::ifstream::in | std::ifstream::binary);
		if (in_mean.fail()) {
			cout << "open file failed! filename: " << (opt.dataPath + "/data_mean.bin").c_str() << endl;
			return;
		}
		MTYPE* meanData = (MTYPE*) malloc (3072*sizeof(MTYPE));
		for (int j = 0; j < 3072; j++)
			in_mean.read((char*)(meanData+j), sizeof(MTYPE));
		in_mean.close();

		int unit = (32 - opt.imSize) / 2;
/*
		int startX[10] = {0, 2*unit, unit, 0, 2*unit, 0, 2*unit, unit, 0, 2*unit};
		int startY[10] = {0, 0, unit, 2*unit, 2*unit, 0, 0, unit, 2*unit, 2*unit};
		int flip[10] = {0,0,0,0,0, 1,1,1,1,1};
*/


		vector<int> startX(numViews);
		vector<int> startY(numViews);
		vector<int> flip(numViews);
		for (int i = 0; i < numViews; i++) {
			startX[i] = rand() % (2*unit + 1);
			startY[i] = rand() % (2*unit + 1);
			flip[i] = rand() % 2;
		}
		//startX[0] = unit; startY[0] = unit;
		//startX[numViews/2] = unit; startY[numViews/2] = unit;

		/*
		int startX[2] = {unit, unit};
		int startY[2] = {unit, unit};
		int flip[2] = {0,1};
		*/


		/*
		vector<int> flip(numViews);
		for (int i = 0; i < numViews ; i++)
			flip[i] = rand() % 2;
		 */
		int meanStartX = unit;
		int meanStartY = unit;

		for (int batch = 0; batch < CPUData.size(); batch++) {
			int batchSize = CPUData[batch]->getNumCols();
			tmp.resize(opt.imSize*opt.imSize*opt.numChannels, batchSize);
			MTYPE* destData = tmp.getData();
			MTYPE* srcData = CPUData[batch]->getData();
			for (int r = 0; r < numViews; r++) {
				for (int l = 0; l < batchSize; l++) {
					for (int i = 0; i < opt.imSize; i++)
						for (int j = 0; j < opt.imSize; j++)
							for (int k = 0; k < opt.numChannels; k++) {
								destIdx = ((k*opt.imSize + j) * opt.imSize + i) * batchSize + l;
								if (flip[r] == 0)
									srcIdx = ((k*32 + j + startY[r]) * 32 + i + startX[r]) * batchSize + l;
								else
									srcIdx = ((k*32 + j + startY[r]) * 32 + (opt.imSize - 1 - i) + startX[r]) * batchSize + l;
								meanIdx = (k*32 + j + meanStartY) * 32 + i + meanStartX;
								destData[destIdx] = srcData[srcIdx] - meanData[meanIdx];
							}
				}
				GPUData[batch*numViews+r]->copyFromHost(tmp, true);
			}
		}
	}
	else {
		Matrix tmp;
		int destIdx, srcIdx;

		int unit = (32 - opt.imSize) / 2;
		vector<int> startX(numViews);
		vector<int> startY(numViews);

		for (int i = 1; i < numViews; i++) {
			startX[i] = rand() % (2*unit + 1);
			startY[i] = rand() % (2*unit + 1);
		}
		startX[0] = unit; startY[0] = unit;
		startX[numViews/2] = unit; startY[numViews/2] = unit;


		int flip[10] = {0,0,0,0,0, 1,1,1,1,1};
		/*
		vector<int> flip(numViews);
		for (int i = 0; i < numViews ; i++)
			flip[i] = rand() % 2;
		*/

		for (int batch = 0; batch < CPUData.size(); batch++) {
			int batchSize = CPUData[batch]->getNumCols();
			tmp.resize(opt.imSize*opt.imSize*opt.numChannels, batchSize);
			MTYPE* destData = tmp.getData();
			MTYPE* srcData = CPUData[batch]->getData();
			for (int r = 0; r < numViews; r++) {
				for (int l = 0; l < batchSize; l++) {
					for (int i = 0; i < opt.imSize; i++)
						for (int j = 0; j < opt.imSize; j++)
							for (int k = 0; k < opt.numChannels; k++) {
								destIdx = ((k*opt.imSize + j) * opt.imSize + i) * batchSize + l;
								if (flip[r] == 0)
									srcIdx = ((k*32 + j + startY[r]) * 32 + i + startX[r]) * batchSize + l;
								else
									srcIdx = ((k*32 + j + startY[r]) * 32 + (opt.imSize - 1 - i) + startX[r]) * batchSize + l;
								destData[destIdx] = srcData[srcIdx];
							}
				}
				GPUData[batch*numViews+r]->copyFromHost(tmp, true);
			}
		}
	}
}

void assembleNVMatrix(vector<NVMatrix>& matrices, NVMatrix& target, int axis) {
	int n = matrices.size();
	assert(n > 0);
	int numRows = matrices[0].getNumRows();
	int numCols = matrices[0].getNumCols();
	int leadingDim = matrices[0].getLeadingDim();
	int followingDim = matrices[0].getFollowingDim();
	bool trans = matrices[0].isTrans();
	target.setTrans(trans);
	if (axis == 0)
		target.resize(numRows*n, numCols);
	else
		target.resize(numRows, numCols*n);

	float* srcData;
	float* destData = target.getDevData();

	for (int i = 0; i < matrices.size(); i++) {
		assert(matrices[i].getNumRows() == numRows && matrices[i].getNumCols() == numCols && matrices[i].isTrans() == trans);
		srcData = matrices[i].getDevData();
		kAssemble <<<256, 256>>> (destData, srcData, i, leadingDim, followingDim, n, axis, trans);
		cutilCheckMsg("assembleNVMatrix: Kernel execution failed");
	}
}

void assembleNVMatrix(NVMatrix& mat1, NVMatrix& mat2, NVMatrix& target, int axis) {
	int r1 = mat1.getNumRows(), r2 = mat2.getNumRows();
	int c1 = mat1.getNumCols(), c2 = mat2.getNumCols();
	bool trans = mat1.isTrans();
	assert(trans == mat2.isTrans());
	int l1 = mat1.getLeadingDim(), l2 = mat2.getLeadingDim();
	int f1 = mat1.getFollowingDim(), f2 = mat2.getFollowingDim();
	target.setTrans(trans);
	if (axis == 0) {
		assert(c1 == c2);
		target.resize(r1+r2, c1);
	}
	else {
		assert(r1 == r2);
		target.resize(r1, c1+c2);
	}

	float* src1 = mat1.getDevData(), *src2 = mat2.getDevData();
	float* dest = target.getDevData();

	kAssemble <<<256, 256>>> (dest, src1, src2, l1, f1, l2, f2, axis, trans);
	cutilCheckMsg("assembleNVMatrix: Kernel execution failed");
}

void splitNVMatrix(vector<NVMatrix>& targets, NVMatrix& mat, int axis) {
	int n = targets.size();
	assert(n > 0);
	int numRows = mat.getNumRows();
	int numCols = mat.getNumCols();
	if (axis == 0) assert(numRows % n == 0);
	else assert(numCols % n == 0);
	int leadingDim;
	int followingDim;
	bool trans = mat.isTrans();
	float* srcData = mat.getDevData();
	float* destData;

	for (int i = 0; i < n; i++) {
		if (axis == 0) targets[i].resize(numRows/n, numCols);
		else targets[i].resize(numRows, numCols/n);
		targets[i].setTrans(trans);
		leadingDim = targets[i].getLeadingDim();
		followingDim = targets[i].getFollowingDim();
		destData = targets[i].getDevData();
		kSplit <<<256, 256>>> (srcData, destData, i, leadingDim, followingDim, n, axis, trans);
		cutilCheckMsg("assembleNVMatrix: Kernel execution failed");
	}
}

void splitNVMatrix(NVMatrix& t1, NVMatrix& t2, NVMatrix& mat, int n1, int n2, int axis) {
	int numRows = mat.getNumRows();
	int numCols = mat.getNumCols();

	if (axis == 0) {
		assert(n1+n2 == numRows);
		t1.resize(n1, numCols);
		t2.resize(n2, numCols);
	}
	else {
		assert(n1+n2 == numCols);
		t1.resize(numRows, n1);
		t2.resize(numRows, n2);
	}
	bool trans = mat.isTrans();
	t1.setTrans(trans);
	t2.setTrans(trans);

	int l1 = t1.getLeadingDim(), f1 = t1.getFollowingDim();
	int l2 = t2.getLeadingDim(), f2 = t2.getFollowingDim();

	float* src = mat.getDevData();
	float* dest1 = t1.getDevData(), *dest2 = t2.getDevData();

	kSplit <<<256, 256>>> (src, dest1, dest2, l1, f1, l2, f2, axis, trans);
	cutilCheckMsg("assembleNVMatrix: Kernel execution failed");
}

void genFilterMask(NVMatrix& target, int numRows, int numCols, MTYPE prob, curandState* devStates) { // prob is the probability of update
	target.resize(numRows, numCols);
	target.setTrans(false);
	MTYPE* data = target.getDevData();
	kFilterMask<<<numCols, 256>>>(data, numRows, numCols, prob, devStates);
}

void genRandBinMatrix(NVMatrix& target, int numRows, int numCols, MTYPE prob, curandState* devStates) { // prob is the probability of update
	target.resize(numRows, numCols);
	target.setTrans(false);
	MTYPE* data = target.getDevData();
	kRandBinMat<<<256, 256>>>(data, numRows, numCols, prob, devStates);
}

void genRandBinMatrix(NVMatrix& target, NVMatrix& like, MTYPE prob, curandState* devStates) { // prob is the probability of update
	target.resize(like.getNumRows(), like.getNumCols());
	target.setTrans(like.isTrans());
	MTYPE* data = target.getDevData();
	kRandBinMat<<<256, 256>>>(data, like.getNumRows(), like.getNumCols(), prob, devStates);
}

curandState* init_cuda_rand(int len) {
	/*
	int* seeds = (int*) malloc(len*sizeof(int));
	for (int i = 0; i < len; i++)
		seeds[i] = rand();
	int* seedsDev;
	CUDA_CALL(cudaMalloc((void**)&seedsDev, len*sizeof(int)));
	CUDA_CALL(cudaMemcpy(seedsDev, seeds, len*sizeof(int), cudaMemcpyHostToDevice));
	*/
	assert(len % 256 == 0);
	int seed = rand();
	curandState *devStates;
	CUDA_CALL(cudaMalloc((void**)&devStates, len*sizeof(curandState)));
	kRandSetup<<<len/256, 256>>>(devStates, seed);
	return devStates;
}

/*
 * maxout operation. mask is the indicator about who got the max
 */
void convMaxOut(NVMatrix& image, NVMatrix& target, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!image.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(image.getNumRows() == numPixels * numColors);
	assert(image.getNumCols() == numCases);

	target.resize(numPixels * numGroups, numCases);
	target.setTrans(false);
	float* data_in = image.getDevData();
	float* data_out = target.getDevData();
	kMaxOut<<<numPixels, numCases>>>(data_in, data_out, poolSize, poolStride, numGroups, numColors);
}

void convMaxOut(NVMatrix& image, NVMatrix& target, NVMask& mask, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!image.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(image.getNumRows() == numPixels * numColors);
	assert(image.getNumCols() == numCases);

	target.resize(numPixels * numGroups, numCases);
	target.setTrans(false);
	mask.resize(target);

	float* data_in = image.getDevData();
	float* data_out = target.getDevData();
	int* data_mask = mask.getDevData();
	kMaxOut<<<numPixels, numCases>>>(data_in, data_out, data_mask, poolSize, poolStride, numGroups, numColors);
}

/*
 * gradient operator for maxout
 */
void convMaxOutUndo(NVMatrix& maxGrad, NVMatrix& target, NVMatrix& image, NVMatrix& maxOut, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!maxGrad.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(maxGrad.getNumRows() == numPixels * numGroups);
	assert(maxGrad.getNumCols() == numCases);

	target.resize(numPixels * numColors, numCases);
	target.setTrans(false);
	float* data_grad = maxGrad.getDevData();
	float* data_target = target.getDevData();
	float* data_image = image.getDevData();
	float* data_max = maxOut.getDevData();
	kMaxOutUndo<<<numPixels, numCases>>>(data_grad, data_target, data_image, data_max, poolSize, poolStride, numGroups, numColors);
}

void convMaxOutUndo(NVMatrix& maxGrad, NVMatrix& target, NVMask& mask, int numColors, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!maxGrad.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(maxGrad.getNumRows() == numPixels * numGroups);
	assert(maxGrad.getNumCols() == numCases);
	assert(mask.getSize() == maxGrad.getNumElements());

	target.resize(numPixels * numColors, numCases);
	target.setTrans(false);
	float* data_grad = maxGrad.getDevData();
	float* data_target = target.getDevData();
	int* data_mask = mask.getDevData();
	kMaxOutUndo<<<numPixels, numCases>>>(data_grad, data_target, data_mask, poolStride, numGroups, numColors);
}
/*
 * hard competition
 */
void convCompeteOut(NVMatrix& image, NVMatrix& target, NVMask& mask, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!image.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(image.getNumRows() == numPixels * numColors);
	assert(image.getNumCols() == numCases);

	target.resize(numPixels * numColors, numCases);
	target.setTrans(false);
	mask.resize(numPixels * numGroups * numCases);

	float* data_in = image.getDevData();
	float* data_out = target.getDevData();
	int* data_mask = mask.getDevData();
	kCompeteOut<<<numPixels, numCases>>>(data_in, data_out, data_mask, poolSize, poolStride, numGroups, numColors);
}

void convCompeteOutUndo(NVMatrix& maxGrad, NVMatrix& target, NVMask& mask, int numColors, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!maxGrad.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(maxGrad.getNumRows() == numPixels * numColors);
	assert(maxGrad.getNumCols() == numCases);
	assert(mask.getSize() == numPixels * numGroups * numCases);

	target.resize(numPixels * numColors, numCases);
	target.setTrans(false);
	float* data_grad = maxGrad.getDevData();
	float* data_target = target.getDevData();
	int* data_mask = mask.getDevData();
	kCompeteOutUndo<<<numPixels, numCases>>>(data_grad, data_target, data_mask, poolStride, numGroups, numColors);
}

void convCompeteAbs(NVMatrix& image, NVMatrix& target, NVMask& mask, int numColors, int poolSize, int poolStride, int imgSizeX, int numCases) {
	//assert(numColors % poolSize == 0);
	assert(!image.isTrans());
	int numGroups = (numColors - 1) / poolStride + 1;
	int numPixels = imgSizeX * imgSizeX;
	assert(image.getNumRows() == numPixels * numColors);
	assert(image.getNumCols() == numCases);

	target.resize(numPixels * numColors, numCases);
	target.setTrans(false);
	mask.resize(numPixels * numGroups * numCases);

	float* data_in = image.getDevData();
	float* data_out = target.getDevData();
	int* data_mask = mask.getDevData();
	kCompeteAbs<<<numPixels, numCases>>>(data_in, data_out, data_mask, poolSize, poolStride, numGroups, numColors);
}


void NVNormalizeCol(NVMatrix& mat, float max_norm) {
	float norm = mat.norm();
	if (norm > max_norm)
		mat.scale(max_norm / norm);
}

void NVNormalizeCol1(NVMatrix& mat, NVMatrix& normCol, NVMatrix& tmp, NVMatrix& div, float max_norm) {
	//NVMatrix normCol, tmp;
	mat.eltwiseMult(mat, tmp);
	tmp.sum(0, normCol);
	normCol.pow(0.5f);
	normCol.scale(1.0/max_norm, div);
	div.maxWithScalar(1.0f);
	mat.eltwiseDivideByVector(div);
}

void NVNormalizeCol2(NVMatrix& mat, NVMatrix& bias, NVMatrix& normCol, NVMatrix& tmp, float max_norm) {
	//NVMatrix normCol, tmp;
	mat.eltwiseMult(mat, tmp);
	tmp.sum(0, normCol);
	normCol.pow(0.5f);
	float maxColNorm = normCol.max();
	if (maxColNorm > max_norm) {
		mat.scale(max_norm / maxColNorm);
		bias.scale(max_norm / maxColNorm);
	}
}

void scaleWeight(string fileName, float scale, int numRows, int numCols) {
	NVMatrix weight;
	weight.resize(numRows, numCols);
	NVReadFromFile(weight, fileName);
	weight.scale(scale);
	NVSaveToFile(weight, fileName);
}


void scaleWeights5(string dirName, float scale) {
	extern LayerOpt opt1, opt2, opt3, opt4, optTop;
	scaleWeight(dirName + "/weight1.bin", scale, opt1.numVis, opt1.numFilters);
	scaleWeight(dirName + "/weight2.bin", scale, opt2.numVis, opt2.numFilters);
	if (strcmp(opt3.layerType, "local") == 0)
		scaleWeight(dirName + "/weight3.bin", scale, opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters);
	else if (strcmp(opt3.layerType, "conv") == 0)
		scaleWeight(dirName + "/weight3.bin", scale, opt3.numVis, opt3.numFilters);
	scaleWeight(dirName + "/weight4.bin", scale, opt4.numVis, opt4.numFilters);
	scaleWeight(dirName + "/weightTop.bin", scale, optTop.numVis, optTop.numFilters);

	scaleWeight(dirName + "/bias1.bin", scale, opt1.numFilters, 1);
	scaleWeight(dirName + "/bias2.bin", scale, opt2.numFilters, 1);
	if (strcmp(opt3.layerType, "local") == 0)
		scaleWeight(dirName + "/bias3.bin", scale, opt3.numFilters * opt3.outX * opt3.outX, 1);
	else if (strcmp(opt3.layerType, "conv") == 0)
		scaleWeight(dirName + "/bias3.bin", scale, opt3.numFilters, 1);
	scaleWeight(dirName + "/bias4.bin", scale, 1, opt4.numFilters);
	scaleWeight(dirName + "/biasTop.bin", scale, 1, optTop.numFilters);

}

void computeThresQuadraticGrad(NVMatrix& labels, NVMatrix& act, NVMatrix& actGrad)
{}





