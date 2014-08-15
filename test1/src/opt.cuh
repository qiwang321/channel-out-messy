#ifndef OPT_CUH_
#define OPT_CUH_

#include <nvmatrix.cuh>
#include <matrix.cuh>
#include <string>

using namespace std;

typedef struct LayerOpt{
	char testName[20];
	char layerName[20];
	char* trainData;
	char* labelFile; // specifying the labels of the data
	int labelSize; // number of classes we want to classify
	int imSize;
	int patchSize;
	int numChannels;
	int numFilters;
	int paddingStart; // start position of the padded image

	int batchSize; // minibatchsize
	int numTrain;
	int numTest;
	int numEpochs;

	MTYPE initstv;
	MTYPE mom ;
	MTYPE lrW;
	MTYPE lrB;
	MTYPE weightDecay;
	//temporarily no sparse constraint
	MTYPE sparseParam;
	MTYPE sparseWeight;

	int poolSize;
	int poolStride;
	int poolStartX;
	int outX;
	int poolOutX;
	char pooler[10];

	int numVis; // number of visible units, used for the softmax layer

	char neuronType[10];
	char layerType[10];

	// these parameters are for pretraining
	int patchNum;
	int batchSize_pre;
	int batchNum_pre;
	int numEpochs_pre;
	MTYPE initstv_pre;
	MTYPE mom_pre;
	MTYPE lrW_pre;
	MTYPE lrB_pre;

	MTYPE weightDecay_pre;
	char neuronType_pre[10];
	// side parameters
	int numFilters_side;
	// rnorm paramters
	int sizeF;
	MTYPE addScale;
	MTYPE powScale;

	int partialSum;
	int numViews;

	bool loadParam;
	bool whitened;
	int maxOutPoolSize;
	int maxOutPoolStride;
	int numGroups;
	string dataPath;
	string weightPath;
	string exp;

	// weight regularization
	float maxNorm;
	// learning rate decay
	char momIncType[15];
	float momStartScale;
	float momIncFactor;
	float momMaxRate;
	char lrDecayType[15];
	float lrStartScale;
	float lrDecayFactor;
	float lrMinRate;

	float inputScale;

	bool flip; // whether to flip the training images

	float keepStartRate;
	float keepIncRate;
	float keepEndRate;
	float keepInputStartRate;
	float keepInputEndRate;
	float keepInputIncRate;

}LayerOpt;

typedef struct Dim{
	int dataX;
	int dataY;
	int dataC;
	int batchSize;
	int numBatches;

	int filterX;
	int numFilters;
	int stride; // stride of the filters
	int padding; // zero padding of the image

	int poolSize;
	int poolStride;
	int poolStartX;
	int poolOutX;
	char pooler[10];
}Dim;




#endif
