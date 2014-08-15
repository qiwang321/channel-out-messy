/*
 * train_filter.cu
 *
 *  Created on: Jul 3, 2013
 *      Author: qwang37
 */

#include <iostream>
#include <fstream>
#include <string.h>

#include <matrix.cuh>
#include <nvmatrix.cuh>
#include <cudaconv2.cuh>
//#include <matrix_funcs.h>
using namespace std;

int main() {
	//printf("program start...\n");
	//MTYPE* data = sampleImages(500000, 5);


	char* dataFile = "/scratch0/qwang37/cifar-10-batches-bin/cifar_patches.bin";
	ifstream in;
	in.open(dataFile);
	if (in.fail()) {
		printf("input file open failed!\n");
		exit(-1);
	}
	
	int patchSize = 5;
	int numChannels = 3;
	int numFilters = 64;

	int batchSize = 512; // minibatchsize
	int batchNum = 2000;
	int numEpochs = 100;

	int numRows = 64;
	int numCols = patchSize*patchSize*numChannels;
	NVMatrix weightForward(numRows, numCols);
	NVMatrix weightBackward(numRows, numCols);
	in.open(dataFile);



	/*MTYPE* data = (MTYPE*) malloc(200*3072*sizeof(MTYPE));
	char* labels = (char*) malloc(200*sizeof(char));
	for (int i = 0; i < 200; i++) {
		labels[i] = in.get();
		for (int j = 0; j < 3072; j++) {
			data[i*3072+j] = MTYPE(in.get());
		}
	}*/
	//Matrix *m1 = new Matrix(data, 200, 3072);

	printf("images\n");
	MTYPE* data = (MTYPE*) malloc(8*3*3*2*sizeof(MTYPE));
	for (int i = 1; i <= 8*3*3*2; i++) {
		data[i-1] = i; // first image is a bunch of "0"s; second image is a bunch of "1"s
	}
	Matrix *im = new Matrix(data, 8*3*3, 2, true);
	NVMatrix *nvim = new NVMatrix(*im, true);
 
	printf("filters\n");
	MTYPE* data2 = (MTYPE*) malloc(32*4*2*sizeof(MTYPE));
	for (int i = 1; i <= 32*4*2; i++) {
		data2[i-1] = i; // filters are a bunch of "1"s
	}
	Matrix *f = new Matrix(data2, 32*4, 2, true);
	NVMatrix *nvf = new NVMatrix(*f, true);

	NVMatrix *targets = new NVMatrix();
	
	convWeightActs(*nvim, *nvf, *targets, 3, 2, 2, 2, 0, 1, 8, 2, 0);
	printf("numRows: %d, numCols: %d\n", targets->getNumRows(), targets->getNumCols());
	targets->print(targets->getNumRows(), targets->getNumCols());
	
}

