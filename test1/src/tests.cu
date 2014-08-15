#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <time.h>

#include <nvmatrix.cuh>
#include <cudaconv2.cuh>
#include "routines.cuh"
#include <layer_kernels.cuh>

using namespace std;

extern LayerOpt opt1, opt2, opt3, opt4, optTop;
extern FILE* pFile;
//void assignOpt();

// test blocks
void testSamplePatches() {
	printf("testing samplePatches()...\n");
	int patchSize = 16;
	int numPatches = 10000;
	int dims[3] = {32, 32, 3};
	int numRecords = 1000;
	char* in_name = "/scratch0/qwang37/cifar-10-batches-bin/cifar_normalized.bin";
	char* out_name1 = "/scratch0/qwang37/cifar-10-batches-bin/cifar_samplePatch_test_half.bin";
	samplePatches(patchSize, numPatches, dims, numRecords, in_name, out_name1);
	patchSize = 32;
	char* out_name2 = "/scratch0/qwang37/cifar-10-batches-bin/cifar_samplePatch_test_full.bin";
	samplePatches(patchSize, numPatches, dims, numRecords, in_name, out_name2);
	printf("samplePatches() test done!\n");
}

void testMult() {
	printf("rightMult test\n");
	MTYPE data1[6] = {0,1,2,3,4,5};
	Matrix m1(data1, 2, 3, false);
	NVMatrix nvm1(m1, true);

	MTYPE data2[9] = {0,1,2,3,4,5,6,7,8};
	Matrix m2(data2, 3, 3, false);
	NVMatrix nvm2(m2, true);

	NVMatrix nvm3;
	nvm1.rightMult(nvm2,1);

	nvm1.printShape("nvm1");
	printf("nvm1 _isTrans = %d\n", nvm1.isTrans());
	nvm1.print(nvm1.getNumRows(), nvm1.getNumCols());

}

void testConv() {
	printf("convolution test\n");
	MTYPE* data = (MTYPE*) malloc(8*3*3*2*sizeof(MTYPE));
	for (int i = 1; i <= 8*3*3*2; i++) {
		data[i-1] = i; // first image is a bunch of "0"s; second image is a bunch of "1"s
	}
	Matrix *im = new Matrix(data, 8*3*3, 2, true); // the transpose specification does not have effect here!
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

void testMatrixIO() {
	printf("testing Matrix IO...\n");
	MTYPE data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	Matrix rm(data, 2, 4, true);
	Matrix cm(data, 4, 2, false);
	NVMatrix NVrm(rm, true);
	NVMatrix NVcm(cm, true);
	NVSaveToFile(NVrm, "NVrm.bin");
	NVSaveToFile(NVcm, "NVcm.bin");
	/* attention: the Matrix and NVMatrix classes do not
	have proper direct copying assignment operators!
	assignment has to be done through reference
	 */
	NVMatrix NVrm1(2,4), NVcm1(4,2);
	NVrm1.setTrans(true); NVcm1.setTrans(false);
	NVReadFromFile(NVrm1, "NVrm.bin");
	NVReadFromFile(NVcm1, "NVcm.bin");
	NVrm1.printShape("NVrm1");
	NVrm1.print(NVrm1.getNumRows(), NVrm1.getNumCols());
	NVcm1.printShape("NVcm1");
	NVcm1.print(NVcm1.getNumRows(), NVcm1.getNumCols());
	printf("Matrix IO test complete!\n");
}

void testDataIO() {
	printf("testing Data IO...\n");
	MTYPE data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	Matrix rm(data, 2, 4, true);
	Matrix cm(data, 4, 2, false);
	NVMatrix NVrm(rm, true);
	NVMatrix NVcm(cm, true);
	NVSaveToFile(NVrm, "NVm.bin", true);
	NVSaveToFile(NVcm, "NVm.bin", true);
	/* attention: the Matrix and NVMatrix classes do not
	have proper direct copying assignment operators!
	assignment has to be done through reference
	 */
	NVMatrix NVrm1(2,4), NVcm1(4,2);
	NVrm1.setTrans(true); NVcm1.setTrans(false);
	NVReadFromFile(NVrm1, "NVm.bin");
	NVReadFromFile(NVcm1, "NVm.bin", 2);
	NVrm1.printShape("NVrm1");
	NVrm1.print(NVrm1.getNumRows(), NVrm1.getNumCols());
	NVcm1.printShape("NVcm1");
	NVcm1.print(NVcm1.getNumRows(), NVcm1.getNumCols());
	printf("Data IO test complete!\n");
}

/*
void testTrainFCAE() {
	printf("testing trainFCAE()...\n");
	LayerOpt opt1;
	char* layerName = "layer1";
	char* df = "/scratch0/qwang37/cifar-10-batches-bin/cifar_patches.bin";
	opt1.layerName = layerName;
	opt1.dataFile = df;
	opt1.patchSize = 5;
	opt1.numChannels = 3;
	opt1.numFilters = 64;
	opt1.batchSize = 2000;
	opt1.batchNum = 1000;
	opt1.numEpochs = 100;
	opt1.initstv = 0.01;
	opt1.mom = 0.0;
	opt1.lrW = 0.01;
	opt1.lrB = 0.01;
	opt1.weightDecay = 0.003;
	opt1.sparseParam = 0.035;
	opt1.sparseWeight = 0.0;
	NVMatrix weight1; NVMatrix bias1; // parameters for the first layer
	char* weightFile ="/scratch0/qwang37/cifar-10-batches-bin/filters5x5_layer1.bin";
	char* biasFile = "/scratch0/qwang37/cifar-10-batches-bin/biases5x5_layer1.bin";
	trainFCAE(opt1, weight1, bias1, weightFile, biasFile);
	//weight1.printShape("forward weight");
	//weight1.print(weight1.getNumRows(), weight1.getNumCols());
	printf("trainFCAE() test complete!\n");
}
 */

/*
void testGenerateDataConv(char* poolType) {
	printf("testing testGenerateDataConv(%s)...\n", poolType);
	Dim dims;
	dims.dataX = 32; dims.dataY = 32; dims.dataC = 3; dims.batchSize = 5000; dims.numBatches = 10;
	dims.filterX = 5; dims.numFilters = 64; dims.stride = 1; dims.padding = 0;
	dims.poolSize = 3; dims.poolStride = 2; dims.poolStartX = 0; strcpy(dims.pooler, poolType);
	dims.poolOutX = (dims.dataX - dims.filterX + 1 - dims.poolSize) / dims.poolStride + 1;

	char* sourceFile = "/scratch0/qwang37/cifar-10-batches-bin/cifar_normalized.bin";
	char* destFile = (char*) malloc (100);
	strcpy(destFile, "/scratch0/qwang37/cifar-10-batches-bin/cifar_layer2data_conv_");
	strcat(destFile, poolType);
	strcat(destFile, ".bin");
	remove(destFile);

	char* layerType = "conv";
	NVMatrix weight(dims.filterX*dims.filterX*dims.dataC, dims.numFilters), biases(1, dims.numFilters);
	NVReadFromFile(weight, "/scratch0/qwang37/cifar-10-batches-bin/filters5x5_layer1.bin");
	NVReadFromFile(biases, "/scratch0/qwang37/cifar-10-batches-bin/biases5x5_layer1.bin");
	generateData(sourceFile, destFile, layerType, weight, biases, dims);
	printf("testGenerateDataConv() test complete!\n");
}
 */


/*
void testGenerateDataFC() {
	printf("testing testGenerateDataFC()...\n");
	Dim dims;
	dims.dataX = 5; dims.dataY = 5; dims.dataC = 3; dims.batchSize = 100000; dims.numBatches = 5;
	dims.filterX = 5; dims.numFilters = 64; dims.stride = 1; dims.padding = 0;

	char* sourceFile = "/scratch0/qwang37/cifar-10-batches-bin/cifar_patches.bin";
	char* destFile = "/scratch0/qwang37/cifar-10-batches-bin/cifar_layer2data_FC.bin";
	char* layerType = "FC";
	NVMatrix weight(dims.filterX*dims.filterX*dims.dataC, dims.numFilters), biases(1, dims.numFilters);
	NVReadFromFile(weight, "/scratch0/qwang37/cifar-10-batches-bin/filters5x5_layer1.bin");
	NVReadFromFile(biases, "/scratch0/qwang37/cifar-10-batches-bin/biases5x5_layer1.bin");
	remove(destFile);
	generateData(sourceFile, destFile, layerType, weight, biases, dims);
	printf("testGenerateDataFC() test complete!\n");
}
 */

void testNVLabelReadFromFile() {
	printf("testing NVLabelReadFromFile()...\n");
	NVMatrix labels(10,10);
	labels.setTrans(false);
	NVLabelReadFromFile(labels, "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", 101);
	labels.printShape("labels");
	labels.print(10,10);
	printf("NVLabelReadFromFile() test complete!\n");
}

void testNVRawLabelReadFromFile() {
	printf("testing NVRawLabelReadFromFile()...\n");
	NVMatrix labels(1,10);
	NVRawLabelReadFromFile(labels, "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", 101);
	labels.printShape("labels");
	labels.print(1,10);
	printf("NVRawLabelReadFromFile() test complete!\n");
}



void finetune_rnorm() {
	////assignOpt();
	printf("starting finetune_rnorm()!\n");
	fprintf(pFile, "starting finetune_rnorm!\n");

	// initialize cublas
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cublasInit();

	// data and parameters storage
	NVMatrix act1, act1Pool, act1PoolNorm, act1Denom;
	NVMatrix act2, act2Norm, act2NormPool, act2Denom;
	NVMatrix act3;
	NVMatrix act4;
	NVMatrix actTop;
	NVMatrix act1Grad, act1PoolGrad, act1PoolNormGrad;
	NVMatrix act2Grad, act2NormGrad, act2NormPoolGrad;
	NVMatrix act3Grad;
	NVMatrix act4Grad;
	NVMatrix actTopGrad;

	NVMatrix weight1, weight2, weight3, weight4, weightTop;
	NVMatrix weight1Grad, weight2Grad, weight3Grad, weight4Grad, weightTopGrad;
	NVMatrix weight1Inc, weight2Inc, weight3Inc, weight4Inc, weightTopInc;
	NVMatrix weight1GradTmp, weight2GradTmp, weight3GradTmp, weight4GradTmp, weightTopGradTmp;

	NVMatrix bias1, bias2, bias3, bias4, biasTop; // bias4 is just an all-zero dummy vector
	NVMatrix bias1Grad, bias2Grad, bias3Grad, bias4Grad, biasTopGrad;
	NVMatrix bias1Inc, bias2Inc, bias3Inc, bias4Inc, biasTopInc;

	// initialize parameters
	if (opt1.loadParam) {
		weight1.resize(opt1.numVis, opt1.numFilters);
		weight2.resize(opt2.numVis, opt2.numFilters);
		weight3.resize(opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters);
		weight4.resize(opt4.numVis * opt4.outX * opt4.outX, opt4.numFilters);
		weightTop.resize(optTop.numVis, optTop.numFilters);

		bias1.resize(opt1.numFilters, 1);
		bias2.resize(opt2.numFilters, 1);
		bias3.resize(opt3.numFilters * opt3.outX * opt3.outX, 1);
		bias4.resize(opt4.numFilters * opt4.outX * opt4.outX, 1);
		biasTop.resize(1, optTop.numFilters);
		biasTop.setTrans(true);


		NVReadFromFile(weight1, "/scratch0/qwang37/cifar-10-batches-bin/weight1.bin");
		NVReadFromFile(weight2, "/scratch0/qwang37/cifar-10-batches-bin/weight2.bin");
		NVReadFromFile(weight3, "/scratch0/qwang37/cifar-10-batches-bin/weight3.bin");
		NVReadFromFile(weight4, "/scratch0/qwang37/cifar-10-batches-bin/weight4.bin");
		NVReadFromFile(weightTop, "/scratch0/qwang37/cifar-10-batches-bin/weightTop.bin");

		NVReadFromFile(bias1, "/scratch0/qwang37/cifar-10-batches-bin/bias1.bin");
		NVReadFromFile(bias2, "/scratch0/qwang37/cifar-10-batches-bin/bias2.bin");
		NVReadFromFile(bias3, "/scratch0/qwang37/cifar-10-batches-bin/bias3.bin");
		NVReadFromFile(bias4, "/scratch0/qwang37/cifar-10-batches-bin/bias4.bin");
		NVReadFromFile(biasTop, "/scratch0/qwang37/cifar-10-batches-bin/biasTop.bin");
	}
	else {
		initWeights(weight1, opt1.numVis, opt1.numFilters, false, opt1.initstv);
		initWeights(weight2, opt2.numVis, opt2.numFilters, false, opt2.initstv);
		initWeights(weight3, opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters, false, opt3.initstv);
		initWeights(weight4, opt4.numVis * opt4.outX * opt4.outX, opt4.numFilters, false, opt4.initstv);
		initWeights(weightTop, optTop.numVis, optTop.numFilters, false, optTop.initstv);

		initWeights(bias1, opt1.numFilters, 1, false, 0.0);
		initWeights(bias2, opt2.numFilters, 1, false, 0.0);
		initWeights(bias3, opt3.numFilters * opt3.outX * opt3.outX, 1, false, 0.0);
		initWeights(bias4, opt4.numFilters * opt4.outX * opt4.outX, 1, false, 0.0);
		initWeights(biasTop, 1, optTop.numFilters, true, 0.0);
	}

	initWeights(weight1Inc, opt1.numVis, opt1.numFilters, false, 0.0); initWeights(weight1Grad, opt1.numVis, opt1.numFilters, false, 0.0);
	initWeights(weight2Inc, opt2.numVis, opt2.numFilters, false, 0.0); initWeights(weight2Grad, opt2.numVis, opt2.numFilters, false, 0.0);
	initWeights(weight3Inc, opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters, false, 0.0); initWeights(weight3Grad, opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters, false, 0.0); // not useful for 3 and 4
	initWeights(weight4Inc, opt4.numVis * opt4.outX * opt4.outX, opt4.numFilters, false, 0.0); initWeights(weight4Grad, opt4.numVis * opt4.outX * opt4.outX, opt4.numFilters, false, 0.0);
	initWeights(weightTopInc, optTop.numVis, optTop.numFilters, false, 0.0); initWeights(weightTopGrad, optTop.numVis, optTop.numFilters, false, 0.0);

	initWeights(bias1Inc, opt1.numFilters, 1, false, 0.0); initWeights(bias1Grad, opt1.numFilters, 1, false, 0.0);
	initWeights(bias2Inc, opt2.numFilters, 1, false, 0.0); initWeights(bias2Grad, opt2.numFilters, 1, false, 0.0);
	initWeights(bias3Inc, opt3.numFilters * opt3.outX * opt3.outX, 1, false, 0.0); initWeights(bias3Grad, opt3.numFilters * opt3.outX * opt3.outX, 1, false, 0.0); // not useful for 3
	initWeights(bias4Inc, opt4.numFilters * opt4.outX * opt4.outX, 1, false, 0.0); initWeights(bias4Grad, opt4.numFilters * opt4.outX * opt4.outX, 1, false, 0.0);
	initWeights(biasTopInc, 1, opt1.labelSize, true, 0.0); initWeights(biasTopGrad, 1, opt1.labelSize, true, 0.0);


	// read data to host memory (and labels to the GPU memory)
	int imPixels = 32*32*opt1.numChannels;
	int batchSize = opt1.batchSize;
	int trainBatchNum = opt1.numTrain / batchSize;
	int testBatchNum = opt1.numTest / batchSize;
	vector<Matrix*> CPUTrain(trainBatchNum), CPUTest(testBatchNum);
	vector<NVMatrix*> GPUTrain(trainBatchNum), GPUTest(testBatchNum);
	vector<NVMatrix*> GPURawLabelTrain(trainBatchNum), GPURawLabelTest(testBatchNum);

	for (int batch = 0; batch < trainBatchNum; batch++) {
		CPUTrain[batch] = new Matrix(imPixels, batchSize);
		CPUTrain[batch]->setTrans(false);
		GPUTrain[batch] = new NVMatrix();
		hmReadFromFile(*CPUTrain[batch], opt1.dataPath + "/cifar_raw.bin", batch*batchSize);
		GPURawLabelTrain[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTrain[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTrain[batch], opt1.dataPath + "/cifar_labels.bin", batch*batchSize);
	}
	batchSize = opt1.numTrain % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTrain.push_back(new Matrix(imPixels, batchSize));
		CPUTrain.back()->setTrans(false);
		GPUTrain.push_back(new NVMatrix());
		hmReadFromFile(*CPUTrain.back(), opt1.dataPath + "/cifar_raw.bin", trainBatchNum*batchSize);
		GPURawLabelTrain.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTrain.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTrain.back(), opt1.dataPath + "/cifar_labels.bin", trainBatchNum*batchSize);
	}
	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		GPUTest[batch] = new NVMatrix();
		hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar_raw.bin", opt1.numTrain+batch*batchSize);
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+batch*batchSize);
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		GPUTest.push_back(new NVMatrix());
		hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar_raw.bin", opt1.numTrain+testBatchNum*batchSize);
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
	}

	NVMatrix trueLabelLogProbs;
	NVMatrix correctProbs;
	MTYPE cost; // as before, we trace the performance using the cost variable
	MTYPE cost1;
	NVMatrix absM;
	MTYPE weightAbs1, weightAbs2, weightAbs3, weightAbs4, weightAbsTop;
	MTYPE biasAbs1, biasAbs2, biasAbs3, biasAbs4, biasAbsTop;
	MTYPE weightGradAbs1, weightGradAbs2, weightGradAbs3, weightGradAbs4, weightGradAbsTop;
	MTYPE biasGradAbs1, biasGradAbs2, biasGradAbs3, biasGradAbs4, biasGradAbsTop;
	clock_t startClock;
	clock_t tick;

	float lr_scale = 1.0, mom_scale = 1.0;

	cropDataProvider(CPUTest, GPUTest, opt1, true, opt1.whitened); // test data is fixed

	for (int epoch = 0; epoch < opt1.numEpochs; epoch++) {
		cost = 0;
		cost1 = 0;
		cropDataProvider(CPUTrain, GPUTrain, opt1, false, opt1.whitened); // copy data to the GPU side
		cudaThreadSynchronize();
		startClock = clock();
		for (int batch = 0; batch < GPUTrain.size(); batch++) {
			batchSize = GPUTrain[batch]->getNumCols();
			// ====forward pass====
			// 0->1
			//cout << "0->1\n";
			//original
			activateConv(*GPUTrain[batch], act1, weight1, bias1, opt1);
			act1.apply(ReluOperator());
			act1Pool.transpose(false);
			convLocalPool(act1, act1Pool, opt1.numFilters, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX, MaxPooler());
			convResponseNormCrossMap(act1Pool, act1Denom, act1PoolNorm, opt1.numFilters, opt1.sizeF, opt1.addScale/opt1.sizeF, opt1.powScale, false);

			// 1->2
			//cout << "1->2\n";
			//original
			activateConv(act1PoolNorm, act2, weight2, bias2, opt2);
			act2.apply(ReluOperator());
			convResponseNormCrossMap(act2, act2Denom, act2Norm, opt2.numFilters, opt2.sizeF, opt2.addScale/opt2.sizeF, opt2.powScale, false);
			act2NormPool.transpose(false);
			convLocalPool(act2Norm, act2NormPool, opt2.numFilters, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX, MaxPooler());

			// 2->3
			//cout << "2->3\n";
			// original
			activateLocal(act2NormPool, act3, weight3, bias3, opt3);
			act3.apply(ReluOperator());

			// 3->4
			//cout << "3->4\n";
			// original
			activateLocal(act3, act4, weight4, bias4, opt4);
			act4.apply(ReluOperator());

			// 4->top
			//cout << "4->top\n";
			actTop.transpose(true);
			actTop.resize(batchSize, opt1.labelSize);
			activate(act4, actTop, weightTop, biasTop, 0, 1);

			//softmax layer
			NVMatrix& max = actTop.max(1);
			actTop.addVector(max, -1);
			actTop.apply(NVMatrixOps::Exp());
			NVMatrix& sum = actTop.sum(1);
			actTop.eltwiseDivideByVector(sum);
			delete &max;
			delete &sum;

			// compute cost
			computeLogregSoftmaxGrad(*GPURawLabelTrain[batch], actTop, actTopGrad, false, 1);
			actTop.transpose(false);
			computeLogregCost(*GPURawLabelTrain[batch], actTop, trueLabelLogProbs, correctProbs); //labelLogProbs:(1, numCases); correctProbs:(1, numCases)
			cost += correctProbs.sum() / batchSize;
			cost1 += trueLabelLogProbs.sum() / batchSize;


			// ====== back pass ======
			// top -> 4, 3, 2, 1
			//cout << "top -> 4, 3, 2, 1";
			// weight update
			NVMatrix& act4T = act4.getTranspose();
			weightTopGrad.addProduct(act4T, actTopGrad, 0, 1);
			biasTopGrad.addSum(actTopGrad, 0, 0, 1);
			delete &act4T;

			// bp
			actTopGrad.transpose(true);
			NVMatrix& weightTopT = weightTop.getTranspose();
			act4Grad.addProduct(actTopGrad, weightTopT, 0, 1);
			delete &weightTopT;

			// 4->3
			//cout << "4->3\n";
			act4Grad.transpose(false); // convert back to row-major
			act4.transpose(false);
			act4Grad.applyBinary(ReluGradientOperator(), act4);

			localWeightActs(act3, act4Grad, weight4Grad, opt4.imSize, opt4.outX, opt4.outX, opt4.patchSize, opt4.paddingStart, 1, opt4.numChannels, 1);
			bias4Grad.addSum(act4Grad, 1, 0, 1);
			localImgActs(act4Grad, weight4, act3Grad, opt4.imSize, opt4.imSize, opt4.outX, opt4.paddingStart, 1, opt4.numChannels, 1);

			// 3->2
			//cout << "3->2\n";
			// original part
			act3Grad.transpose(false); // convert back to row-major
			act3.transpose(false);
			act3Grad.applyBinary(ReluGradientOperator(), act3);
			localWeightActs(act2NormPool, act3Grad, weight3Grad, opt3.imSize, opt3.outX, opt3.outX, opt3.patchSize, opt3.paddingStart, 1, opt3.numChannels, 1);
			bias3Grad.addSum(act3Grad, 1, 0, 1);
			localImgActs(act3Grad, weight3, act2NormPoolGrad, opt3.imSize, opt3.imSize, opt3.outX, opt3.paddingStart, 1, opt3.numChannels, 1);

			// 2->1
			//cout << "2->1\n";
			// original part
			act2NormPoolGrad.transpose(false);
			act2NormPool.transpose(false);
			convLocalMaxUndo(act2Norm, act2NormPoolGrad, act2NormPool, act2NormGrad, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX);
			convResponseNormCrossMapUndo(act2NormGrad, act2Denom, act2, act2Norm, act2Grad, opt2.numFilters, opt2.sizeF, opt2.addScale/opt2.sizeF, opt2.powScale, false, 0, 1);
			act2Grad.applyBinary(ReluGradientOperator(), act2);
			convWeightActs(act1PoolNorm, act2Grad, weight2GradTmp, opt2.imSize, opt2.outX, opt2.outX, opt2.patchSize, opt2.paddingStart, 1, opt2.numChannels, 1, opt2.partialSum);
			weight2GradTmp.reshape(opt2.outX * opt2.outX / opt2.partialSum, opt2.numChannels * opt2.patchSize * opt2.patchSize * opt2.numFilters);
			weight2Grad.addSum(weight2GradTmp, 0, 0, 1);
			weight2Grad.reshape(opt2.numChannels * opt2.patchSize * opt2.patchSize, opt2.numFilters);
			act2Grad.reshape(opt2.numFilters, opt2.outX * opt2.outX * batchSize);
			bias2Grad.addSum(act2Grad, 1, 0, 1);
			act2Grad.reshape(opt2.numFilters * opt2.outX * opt2.outX, batchSize);
			convImgActs(act2Grad, weight2, act1PoolNormGrad, opt2.imSize, opt2.imSize, opt2.outX, opt2.paddingStart, 1, opt2.numChannels, 1);

			// 1->0
			//cout << "1->0\n";
			// original part
			act1PoolNormGrad.transpose(false);
			act1PoolNorm.transpose(false);
			convResponseNormCrossMapUndo(act1PoolNormGrad, act1Denom, act1Pool, act1PoolNorm, act1PoolGrad, opt1.numFilters, opt1.sizeF, opt1.addScale/opt1.sizeF, opt1.powScale, false, 0, 1);
			convLocalMaxUndo(act1, act1PoolGrad, act1Pool, act1Grad, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX);
			act1Grad.applyBinary(ReluGradientOperator(), act1);
			convWeightActs(*GPUTrain[batch], act1Grad, weight1GradTmp, opt1.imSize, opt1.outX, opt1.outX, opt1.patchSize, opt1.paddingStart, 1, opt1.numChannels, 1, opt1.partialSum);
			weight1GradTmp.reshape(opt1.outX * opt1.outX / opt1.partialSum, opt1.numChannels * opt1.patchSize * opt1.patchSize * opt1.numFilters);
			weight1Grad.addSum(weight1GradTmp, 0, 0, 1);
			weight1Grad.reshape(opt1.numChannels * opt1.patchSize * opt1.patchSize, opt1.numFilters);
			act1Grad.reshape(opt1.numFilters, opt1.outX * opt1.outX * batchSize);
			bias1Grad.addSum(act1Grad, 1, 0, 1);
			act1Grad.reshape(opt1.numFilters * opt1.outX * opt1.outX, batchSize);

			// update
			lr_scale = lrDecay(lr_scale, opt1.lrDecayType, opt1.lrDecayFactor, opt1.lrMinRate);
			mom_scale = momInc(mom_scale, opt1.momIncType, opt1.momIncFactor, opt1.momMaxRate);
			updateWeight(weight1Grad, weight1Inc, weight1, opt1, batchSize, lr_scale, mom_scale);
			updateWeight(weight2Grad, weight2Inc, weight2, opt2, batchSize, lr_scale, mom_scale);
			updateWeight(weight3Grad, weight3Inc, weight3, opt3, batchSize, lr_scale, mom_scale);
			updateWeight(weight4Grad, weight4Inc, weight4, opt4, batchSize, lr_scale, mom_scale);
			updateWeight(weightTopGrad, weightTopInc, weightTop, optTop, batchSize, lr_scale, mom_scale);
			updateBias(bias1Grad, bias1Inc, bias1, opt1, batchSize, lr_scale, mom_scale);
			updateBias(bias2Grad, bias2Inc, bias2, opt2, batchSize, lr_scale, mom_scale);
			updateBias(bias3Grad, bias3Inc, bias3, opt3, batchSize, lr_scale, mom_scale);
			updateBias(bias4Grad, bias4Inc, bias4, opt4, batchSize, lr_scale, mom_scale);
			updateBias(biasTopGrad, biasTopInc, biasTop, optTop, batchSize, lr_scale, mom_scale);
		} // for (int epoch = 0; epoch < opt1.numEpochs; epoch++)

		cudaThreadSynchronize();
		cost /= CPUTrain.size();
		cost1 /= CPUTrain.size();
		printf("\nfinished epoch %d of %d; classify precision = %f; objective = %f; elapsed time = %f seconds\n", epoch, opt1.numEpochs,
				cost, cost1, (float)(clock() - startClock)/CLOCKS_PER_SEC);
		fprintf(pFile, "\nfinished epoch %d of %d; classify precision = %f; objective = %f; elapsed time = %f seconds\n", epoch, opt1.numEpochs,
				cost, cost1, (float)(clock() - startClock)/CLOCKS_PER_SEC);

		/*
		weight1.apply(NVMatrixOps::Abs(), absM);
		weightAbs1 = absM.sum() / absM.getNumElements();
		weight2.apply(NVMatrixOps::Abs(), absM);
		weightAbs2 = absM.sum() / absM.getNumElements();
		weight3.apply(NVMatrixOps::Abs(), absM);
		weightAbs3 = absM.sum() / absM.getNumElements();
		weight4.apply(NVMatrixOps::Abs(), absM);
		weightAbs4 = absM.sum() / absM.getNumElements();
		weightTop.apply(NVMatrixOps::Abs(), absM);
		weightAbsTop = absM.sum() / absM.getNumElements();


		weight1Inc.apply(NVMatrixOps::Abs(), absM);
		weightGradAbs1 = absM.sum() / absM.getNumElements();
		weight2Inc.apply(NVMatrixOps::Abs(), absM);
		weightGradAbs2 = absM.sum() / absM.getNumElements();
		weight3Inc.apply(NVMatrixOps::Abs(), absM);
		weightGradAbs3 = absM.sum() / absM.getNumElements();
		weight4Inc.apply(NVMatrixOps::Abs(), absM);
		weightGradAbs4 = absM.sum() / absM.getNumElements();
		weightTopInc.apply(NVMatrixOps::Abs(), absM);
		weightGradAbsTop = absM.sum() / absM.getNumElements();

		bias1.apply(NVMatrixOps::Abs(), absM);
		biasAbs1 = absM.sum() / absM.getNumElements();
		bias2.apply(NVMatrixOps::Abs(), absM);
		biasAbs2 = absM.sum() / absM.getNumElements();
		bias3.apply(NVMatrixOps::Abs(), absM);
		biasAbs3 = absM.sum() / absM.getNumElements();
		bias4.apply(NVMatrixOps::Abs(), absM);
		biasAbs4 = absM.sum() / absM.getNumElements();
		biasTop.apply(NVMatrixOps::Abs(), absM);
		biasAbsTop = absM.sum() / absM.getNumElements();

		bias1Inc.apply(NVMatrixOps::Abs(), absM);
		biasGradAbs1 = absM.sum() / absM.getNumElements();
		bias2Inc.apply(NVMatrixOps::Abs(), absM);
		biasGradAbs2 = absM.sum() / absM.getNumElements();
		bias3Inc.apply(NVMatrixOps::Abs(), absM);
		biasGradAbs3 = absM.sum() / absM.getNumElements();
		bias4Inc.apply(NVMatrixOps::Abs(), absM);
		biasGradAbs4 = absM.sum() / absM.getNumElements();
		biasTopInc.apply(NVMatrixOps::Abs(), absM);
		biasGradAbsTop = absM.sum() / absM.getNumElements();


		printf("weight abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", weightAbs1, weightAbs2, weightAbs3, weightAbs4, weightAbsTop);
		printf("weight grad abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", weightGradAbs1, weightGradAbs2, weightGradAbs3, weightGradAbs4, weightGradAbsTop);
		printf("bias abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", biasAbs1, biasAbs2, biasAbs3, biasAbs4, biasAbsTop);
		printf("bias grad abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", biasGradAbs1, biasGradAbs2, biasGradAbs3, biasGradAbs4, biasGradAbsTop);

		fprintf(pFile, "weight abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", weightAbs1, weightAbs2, weightAbs3, weightAbs4, weightAbsTop);
		fprintf(pFile, "weight grad abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", weightGradAbs1, weightGradAbs2, weightGradAbs3, weightGradAbs4, weightGradAbsTop);
		fprintf(pFile, "bias abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", biasAbs1, biasAbs2, biasAbs3, biasAbs4, biasAbsTop);
		fprintf(pFile, "bias grad abs: 1--%f, 2--%f, 3--%f, 4--%f, top--%f\n", biasGradAbs1, biasGradAbs2, biasGradAbs3, biasGradAbs4, biasGradAbsTop);
		 */

		// process the test set every 3 epochs
		if (epoch % 3 == 2) {
			cudaThreadSynchronize();
			startClock = clock();
			cost = 0;
			cost1 = 0;
			for (int batch = 0; batch < GPUTest.size(); batch++) {
				batchSize = GPUTest[batch]->getNumCols();
				// ====forward pass====
				// 0->1
				//cout << "0->1\n";
				//original
				activateConv(*GPUTest[batch], act1, weight1, bias1, opt1);
				act1.apply(ReluOperator());
				act1Pool.transpose(false);
				convLocalPool(act1, act1Pool, opt1.numFilters, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX, MaxPooler());
				convResponseNormCrossMap(act1Pool, act1Denom, act1PoolNorm, opt1.numFilters, opt1.sizeF, opt1.addScale/opt1.sizeF, opt1.powScale, false);

				// 1->2
				//cout << "1->2\n";
				//original
				activateConv(act1PoolNorm, act2, weight2, bias2, opt2);
				act2.apply(ReluOperator());
				convResponseNormCrossMap(act2, act2Denom, act2Norm, opt2.numFilters, opt2.sizeF, opt2.addScale/opt2.sizeF, opt2.powScale, false);
				act2NormPool.transpose(false);
				convLocalPool(act2Norm, act2NormPool, opt2.numFilters, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX, MaxPooler());

				// 2->3
				//cout << "2->3\n";
				// original
				activateLocal(act2NormPool, act3, weight3, bias3, opt3);
				act3.apply(ReluOperator());

				// 3->4
				//cout << "3->4\n";
				// original
				activateLocal(act3, act4, weight4, bias4, opt4);
				act4.apply(ReluOperator());

				// 4->top
				//cout << "4->top\n";
				actTop.transpose(true);
				actTop.resize(batchSize, opt1.labelSize);
				activate(act4, actTop, weightTop, biasTop, 0, 1);

				//softmax layer
				NVMatrix& max = actTop.max(1);
				actTop.addVector(max, -1);
				actTop.apply(NVMatrixOps::Exp());
				NVMatrix& sum = actTop.sum(1);
				actTop.eltwiseDivideByVector(sum);
				delete &max;
				delete &sum;

				// compute cost
				computeLogregSoftmaxGrad(*GPURawLabelTest[batch], actTop, actTopGrad, false, 1);
				actTop.transpose(false);
				computeLogregCost(*GPURawLabelTest[batch], actTop, trueLabelLogProbs, correctProbs); //labelLogProbs:(1, numCases); correctProbs:(1, numCases)
				cost += correctProbs.sum() / batchSize;
				cost1 += trueLabelLogProbs.sum() / batchSize;

			} //for (int batch = opt1.batchNum; batch < opt1.batchNum+opt1.testBatchNum; batch++)
			cudaThreadSynchronize();
			cost /= GPUTest.size();
			cost1 /= GPUTest.size();
			printf("\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
					(float)(clock() - startClock)/CLOCKS_PER_SEC);
			fprintf(pFile, "\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
					(float)(clock() - startClock)/CLOCKS_PER_SEC);

			// save checkpoint
			char* weight1File = "/scratch0/qwang37/cifar-10-batches-bin/weight1.bin", *bias1File = "/scratch0/qwang37/cifar-10-batches-bin/bias1.bin";
			char* weight2File = "/scratch0/qwang37/cifar-10-batches-bin/weight2.bin", *bias2File = "/scratch0/qwang37/cifar-10-batches-bin/bias2.bin";
			char* weight3File = "/scratch0/qwang37/cifar-10-batches-bin/weight3.bin", *bias3File = "/scratch0/qwang37/cifar-10-batches-bin/bias3.bin";
			char* weight4File = "/scratch0/qwang37/cifar-10-batches-bin/weight4.bin", *bias4File = "/scratch0/qwang37/cifar-10-batches-bin/bias4.bin";
			char* weightTopFile = "/scratch0/qwang37/cifar-10-batches-bin/weightTop.bin", *biasTopFile = "/scratch0/qwang37/cifar-10-batches-bin/biasTop.bin";

			NVSaveToFile(weight1, weight1File); NVSaveToFile(bias1, bias1File);
			NVSaveToFile(weight2, weight2File); NVSaveToFile(bias2, bias2File);
			NVSaveToFile(weight3, weight3File); NVSaveToFile(bias3, bias3File);
			NVSaveToFile(weight4, weight4File); NVSaveToFile(bias4, bias4File);
			NVSaveToFile(weightTop, weightTopFile); NVSaveToFile(biasTop, biasTopFile);
			printf("Checkpoint saved!\n\n");
			fprintf(pFile, "Checkpoint saved!\n\n");

		} //if (epoch % 10 == 0)

	} // for (int epoch = 0; epoch < opt1.numEpochs; epoch++)
	printf("finetuning_rnorm() complete!\n");
	fprintf(pFile, "finetuning_rnorm() complete!\n");
} // int finetune_rnorm()

void multiViewTest() {
	////assignOpt();
	printf("starting multiViewTest()!\n");
	fprintf(pFile, "starting multiViewTest()!\n");

	// initialize cublas
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cublasInit();

	// data and parameters storage
	NVMatrix act1, act1Pool, act1PoolNorm, act1Denom;
	NVMatrix act2, act2Norm, act2NormPool, act2Denom;
	NVMatrix act3;
	NVMatrix act4;
	NVMatrix actTop;
	NVMatrix softMaxAct;

	NVMatrix weight1, weight2, weight3, weight4, weightTop;
	NVMatrix bias1, bias2, bias3, bias4, biasTop; // bias4 is just an all-zero dummy vector
	// initialize parameters

	weight1.resize(opt1.numVis, opt1.numFilters);
	weight2.resize(opt2.numVis, opt2.numFilters);
	weight3.resize(opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters);
	weight4.resize(opt4.numVis * opt4.outX * opt4.outX, opt4.numFilters);
	weightTop.resize(optTop.numVis, optTop.numFilters);

	bias1.resize(opt1.numFilters, 1);
	bias2.resize(opt2.numFilters, 1);
	bias3.resize(opt3.numFilters * opt3.outX * opt3.outX, 1);
	bias4.resize(opt4.numFilters * opt4.outX * opt4.outX, 1);
	biasTop.resize(1, optTop.numFilters);
	biasTop.setTrans(true);

	NVReadFromFile(weight1, "/scratch0/qwang37/cifar-10-batches-bin/weight1.bin");
	NVReadFromFile(weight2, "/scratch0/qwang37/cifar-10-batches-bin/weight2.bin");
	NVReadFromFile(weight3, "/scratch0/qwang37/cifar-10-batches-bin/weight3.bin");
	NVReadFromFile(weight4, "/scratch0/qwang37/cifar-10-batches-bin/weight4.bin");
	NVReadFromFile(weightTop, "/scratch0/qwang37/cifar-10-batches-bin/weightTop.bin");

	NVReadFromFile(bias1, "/scratch0/qwang37/cifar-10-batches-bin/bias1.bin");
	NVReadFromFile(bias2, "/scratch0/qwang37/cifar-10-batches-bin/bias2.bin");
	NVReadFromFile(bias3, "/scratch0/qwang37/cifar-10-batches-bin/bias3.bin");
	NVReadFromFile(bias4, "/scratch0/qwang37/cifar-10-batches-bin/bias4.bin");
	NVReadFromFile(biasTop, "/scratch0/qwang37/cifar-10-batches-bin/biasTop.bin");

	// read data to host memory (and labels to the GPU memory)
	int imPixels = 32*32*opt1.numChannels;
	int batchSize = opt1.batchSize;
	int testBatchNum = opt1.numTest / batchSize;
	vector<Matrix*> CPUTest(testBatchNum);
	vector<NVMatrix*> GPUTest(testBatchNum*opt1.numViews);
	vector<NVMatrix*> GPURawLabelTest(testBatchNum);

	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		for (int r = 0; r < opt1.numViews; r++)
			GPUTest[batch*opt1.numViews+r] = new NVMatrix();
		hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar_raw.bin", opt1.numTrain+batch*batchSize);
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+batch*batchSize);
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		for (int r = 0; r < opt1.numViews; r++)
			GPUTest.push_back(new NVMatrix());
		hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar_raw.bin", opt1.numTrain+testBatchNum*batchSize);
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
	}

	multiViewDataProvider(CPUTest, GPUTest, opt1, opt1.numViews, opt1.whitened); // copy data to the GPU side

	NVMatrix trueLabelLogProbs;
	NVMatrix correctProbs;
	MTYPE cost; // as before, we trace the performance using the cost variable
	MTYPE cost1;
	clock_t startClock;
	clock_t tick;
	cost = 0;
	cost1 = 0;
	cudaThreadSynchronize();
	startClock = clock();

	for (int batch = 0; batch < CPUTest.size(); batch++) {
		batchSize = CPUTest[batch]->getNumCols();
		for (int r = 0; r < opt1.numViews; r++) {
			// ====forward pass====
			// 0->1
			//cout << "0->1\n";
			//original
			activateConv(*GPUTest[batch*opt1.numViews+r], act1, weight1, bias1, opt1);
			act1.apply(ReluOperator());
			act1Pool.transpose(false);
			convLocalPool(act1, act1Pool, opt1.numFilters, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX, MaxPooler());
			convResponseNormCrossMap(act1Pool, act1Denom, act1PoolNorm, opt1.numFilters, opt1.sizeF, opt1.addScale/opt1.sizeF, opt1.powScale, false);

			// 1->2
			//cout << "1->2\n";
			//original
			activateConv(act1PoolNorm, act2, weight2, bias2, opt2);
			act2.apply(ReluOperator());
			convResponseNormCrossMap(act2, act2Denom, act2Norm, opt2.numFilters, opt2.sizeF, opt2.addScale/opt2.sizeF, opt2.powScale, false);
			act2NormPool.transpose(false);
			convLocalPool(act2Norm, act2NormPool, opt2.numFilters, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX, MaxPooler());

			// 2->3
			//cout << "2->3\n";
			// original
			activateLocal(act2NormPool, act3, weight3, bias3, opt3);
			act3.apply(ReluOperator());

			// 3->4
			//cout << "3->4\n";
			// original
			activateLocal(act3, act4, weight4, bias4, opt4);
			act4.apply(ReluOperator());

			// 4->top
			//cout << "4->top\n";
			actTop.transpose(true);
			actTop.resize(batchSize, opt1.labelSize);
			activate(act4, actTop, weightTop, biasTop, 0, 1);

			//softmax layer
			NVMatrix& max = actTop.max(1);
			actTop.addVector(max, -1);
			actTop.apply(NVMatrixOps::Exp());
			NVMatrix& sum = actTop.sum(1);
			actTop.eltwiseDivideByVector(sum);
			delete &max;
			delete &sum;
			actTop.transpose(false);

			if (r == 0)
				actTop.copy(softMaxAct);
			else
				softMaxAct.add(actTop);
		}// for (r = 0:9)
		softMaxAct.scale(0.1);
		computeLogregCost(*GPURawLabelTest[batch], softMaxAct, trueLabelLogProbs, correctProbs); //labelLogProbs:(1, numCases); correctProbs:(1, numCases)
		cost += correctProbs.sum();
		cost1 += trueLabelLogProbs.sum();
	}//for (batches)

	cudaThreadSynchronize();
	cost /= opt1.numTest;
	cost1 /= opt1.numTest;
	printf("\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
			(float)(clock() - startClock)/CLOCKS_PER_SEC);
	printf("multiViewTest() complete!\n");

	fprintf(pFile, "\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
			(float)(clock() - startClock)/CLOCKS_PER_SEC);
	fprintf(pFile, "multiViewTest() complete!\n");
} // void multiViewTest()


void testCropDataProvider() {
	//assignOpt();
	printf("starting test cropDataProvider()!\n");
	// read data to host memory (and labels to the GPU memory)
	int imPixels = 32*32*opt1.numChannels;
	int batchSize = opt1.batchSize;
	int trainBatchNum = opt1.numTrain / batchSize;
	int testBatchNum = opt1.numTest / batchSize;
	vector<Matrix*> CPUTrain(trainBatchNum), CPUTest(testBatchNum);
	vector<NVMatrix*> GPUTrain(trainBatchNum), GPUTest(testBatchNum);
	vector<NVMatrix*> GPURawLabelTrain(trainBatchNum), GPURawLabelTest(testBatchNum);

	for (int batch = 0; batch < trainBatchNum; batch++) {
		CPUTrain[batch] = new Matrix(imPixels, batchSize);
		CPUTrain[batch]->setTrans(false);
		GPUTrain[batch] = new NVMatrix();
		hmReadFromFile(*CPUTrain[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", batch*batchSize);
		GPURawLabelTrain[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTrain[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTrain[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", batch*batchSize);
	}
	batchSize = opt1.numTrain % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTrain.push_back(new Matrix(imPixels, batchSize));
		CPUTrain.back()->setTrans(false);
		GPUTrain.push_back(new NVMatrix());
		hmReadFromFile(*CPUTrain.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", trainBatchNum*batchSize);
		GPURawLabelTrain.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTrain.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTrain.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", trainBatchNum*batchSize);
	}
	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		GPUTest[batch] = new NVMatrix();
		hmReadFromFile(*CPUTest[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", opt1.numTrain+batch*batchSize);
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", opt1.numTrain+batch*batchSize);
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		GPUTest.push_back(new NVMatrix());
		hmReadFromFile(*CPUTest.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", opt1.numTrain+testBatchNum*batchSize);
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
	}

	cropDataProvider(CPUTrain, GPUTrain, opt1, false, false);
	remove("/scratch0/qwang37/cifar-10-batches-bin/test_out.bin");
	for (int batch = 0; batch < GPUTrain.size(); batch++) {
		NVSaveToFile(*GPUTrain[batch], "/scratch0/qwang37/cifar-10-batches-bin/test_out.bin", true);
	}
	printf("cropDataProvider() test done!\n");
}

void testNVReadFromFileUint8() {
	//assignOpt();
	printf("starting test testNVReadFromFileUint8()!\n");
	// read data to host memory (and labels to the GPU memory)
	int imPixels = 32*32*opt1.numChannels;
	int batchSize = opt1.batchSize;
	int trainBatchNum = opt1.numTrain / batchSize;
	int testBatchNum = opt1.numTest / batchSize;
	vector<Matrix*> CPUTrain(trainBatchNum), CPUTest(testBatchNum);
	vector<NVMatrix*> GPUTrain(trainBatchNum), GPUTest(testBatchNum);
	vector<NVMatrix*> GPURawLabelTrain(trainBatchNum), GPURawLabelTest(testBatchNum);

	for (int batch = 0; batch < trainBatchNum; batch++) {
		CPUTrain[batch] = new Matrix(imPixels, batchSize);
		CPUTrain[batch]->setTrans(false);
		GPUTrain[batch] = new NVMatrix();
		hmReadFromFile(*CPUTrain[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", batch*batchSize);
		GPURawLabelTrain[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTrain[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTrain[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", batch*batchSize);
	}
	batchSize = opt1.numTrain % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTrain.push_back(new Matrix(imPixels, batchSize));
		CPUTrain.back()->setTrans(false);
		GPUTrain.push_back(new NVMatrix());
		hmReadFromFile(*CPUTrain.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", trainBatchNum*batchSize);
		GPURawLabelTrain.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTrain.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTrain.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", trainBatchNum*batchSize);
	}
	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		GPUTest[batch] = new NVMatrix();
		hmReadFromFile(*CPUTest[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", opt1.numTrain+batch*batchSize);
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest[batch], "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", opt1.numTrain+batch*batchSize);
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		GPUTest.push_back(new NVMatrix());
		hmReadFromFile(*CPUTest.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", opt1.numTrain+testBatchNum*batchSize);
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);
		NVRawLabelReadFromFile(*GPURawLabelTest.back(), "/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
	}
	cropDataProvider(CPUTrain, GPUTrain, opt1, false, false);
	remove("/scratch0/qwang37/cifar-10-batches-bin/test_out.bin");
	for (int batch = 0; batch < GPUTrain.size(); batch++) {
		NVSaveToFile(*GPUTrain[batch], "/scratch0/qwang37/cifar-10-batches-bin/test_out.bin", true);
	}
	printf("testNVReadFromFileUint8() test done!\n");
}

void centerData() {
	ifstream in;
	printf("starting centering data\n");
	MTYPE* data = (MTYPE*) malloc(60000*3072*sizeof(MTYPE));
	MTYPE* mean = (MTYPE*) malloc(3072*sizeof(MTYPE));
	char* labels = (char*) malloc(60000*sizeof(char));
	char dir_name[] = "/scratch0/qwang37/cifar-10-batches-bin/";
	char file_name[] = "data_batch_1.bin";
	char full_name[100];
	int record_start;

	for (int j = 0; j < 3072; j++)
		mean[j] = 0.0;

	for (int k = 1; k <= 5; k++) {
		file_name[11] = '0' + k;
		strcpy(full_name, dir_name);
		strcat(full_name, file_name);
		in.open(full_name, std::ifstream::in | std::ifstream::binary);
		if (in.fail()) {
			printf("open data file %d failed!\n", k);
			exit(-1);
		}
		printf("reading batch %d\n", k);

		for (int i = 0; i < 10000; i++) {
			record_start = (k-1)*10000 + i;
			labels[record_start] = in.get();
			for (int j = 0; j < 3072; j++) {
				data[record_start*3072+j] = MTYPE(in.get());
				mean[j] += data[record_start*3072+j];
			}
		}
		in.close();
	}



	char test_name[100];
	strcpy(test_name, dir_name);
	strcat(test_name, "test_batch.bin");
	in.open(test_name, std::ifstream::in | std::ifstream::binary);
	printf("reading test batch\n");

	for (int i = 0; i < 10000; i++) {
		record_start = 5*10000 + i;
		labels[record_start] = in.get();
		for (int j = 0; j < 3072; j++) {
			data[record_start*3072+j] = (MTYPE)in.get();
			//mean[j] += data[record_start*3072+j];
		}
	}
	in.close();

	/*
	for (int j = 0; j < 3072; j++)
		mean[j] /= 50000.0;
	*/
	ifstream in_mean;
	in_mean.open("/scratch0/qwang37/cifar-10-batches-bin/data_mean.bin", std::ifstream::in | std::ifstream::binary);
	if (in_mean.fail()) {
		cout << "open file failed!\n";
		return;
	}
	for (int j = 0; j < 3072; j++)
		in_mean.read((char*)(mean+j), 4);
	in_mean.close();

	for (int i = 0; i < 60000; i++)
		for (int j = 0; j < 3072; j++)
			data[i*3072+j] -= mean[j];
			//data[i*3072+j] -= 127.5;


	ofstream out("/scratch0/qwang37/cifar-10-batches-bin/cifar_centered.bin", std::ofstream::out | std::ifstream::binary);
	out.write((char*)data, 60000*3072*sizeof(MTYPE));
	out.close();
}

void convertToMTYPE() {
	ifstream in;
	printf("starting copying data\n");
	MTYPE* data = (MTYPE*) malloc(60000*3072*sizeof(MTYPE));
	char* labels = (char*) malloc(60000*sizeof(char));
	char dir_name[] = "/scratch0/qwang37/cifar-10-batches-bin/";
	char file_name[] = "data_batch_1.bin";
	char full_name[100];
	int record_start;

	printf("starting copy data\n");
	for (int k = 1; k <= 5; k++) {
		file_name[11] = '0' + k;
		strcpy(full_name, dir_name);
		strcat(full_name, file_name);
		in.open(full_name, std::ifstream::in | std::ifstream::binary);
		if (in.fail()) {
			printf("open data file %d failed!\n", k);
			exit(-1);
		}
		printf("reading batch %d\n", k);

		for (int i = 0; i < 10000; i++) {
			record_start = (k-1)*10000 + i;
			labels[record_start] = in.get();
			for (int j = 0; j < 3072; j++) {
				data[record_start*3072+j] = MTYPE(in.get());
			}
		}
		in.close();
	}

	char test_name[100];
	strcpy(test_name, dir_name);
	strcat(test_name, "test_batch.bin");
	in.open(test_name);
	printf("reading test batch\n");

	for (int i = 0; i < 10000; i++) {
		record_start = 5*10000 + i;
		labels[record_start] = in.get();
		for (int j = 0; j < 3072; j++) {
			data[record_start*3072+j] = (MTYPE)in.get();
		}
	}
	in.close();

	ofstream out("/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin", std::ofstream::out | std::ifstream::binary);
	out.write((char*)data, 60000*3072*sizeof(MTYPE));
	out.close();
}

void testAssembleMatrix() {
	printf("start testAssembleMatrix()\n");
	vector<NVMatrix> matrices(4);
	Matrix tmp(2,3);
	MTYPE* data = tmp.getData();
	for (int i = 0; i < 4; i++)
		matrices[i] = new NVMatrix(2,3);
	// test1
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 6; j++)
			data[j] = i * 6 + j;
		tmp.setTrans(false);
		matrices[i].copyFromHost(tmp, true);
	}
	NVMatrix rowFalse;
	assembleNVMatrix(matrices, rowFalse, 0);
	rowFalse.printShape("rowFalse");
	rowFalse.print(8,3);

	for (int i = 0; i < 4; i++) {
		matrices[i].resize(0,0);
	}
	splitNVMatrix(matrices, rowFalse, 0);
	char a[10];
	for (int i = 0; i < 4; i++) {
		sprintf(a, "rowFalse%d", i);
		matrices[i].printShape(a);
		matrices[i].print(2,3);
	}

	// test2
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 6; j++)
			data[j] = i * 6 + j;
		tmp.setTrans(true);
		matrices[i].copyFromHost(tmp, true);
	}
	NVMatrix rowTrue;
	assembleNVMatrix(matrices, rowTrue, 0);
	rowTrue.printShape("rowTrue");
	rowTrue.print(8,3);

	for (int i = 0; i < 4; i++) {
		matrices[i].resize(0,0);
	}
	splitNVMatrix(matrices, rowTrue, 0);
	for (int i = 0; i < 4; i++) {
		sprintf(a, "rowTrue%d", i);
		matrices[i].printShape(a);
		matrices[i].print(2,3);
	}

	// test3
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 6; j++)
			data[j] = i * 6 + j;
		tmp.setTrans(false);
		matrices[i].copyFromHost(tmp, true);
	}
	NVMatrix colFalse;
	assembleNVMatrix(matrices, colFalse, 1);
	colFalse.printShape("colFalse");
	colFalse.print(2,12);

	for (int i = 0; i < 4; i++) {
		matrices[i].resize(0,0);
	}
	splitNVMatrix(matrices, colFalse, 1);
	for (int i = 0; i < 4; i++) {
		sprintf(a, "colFalse%d", i);
		matrices[i].printShape(a);
		matrices[i].print(2,3);
	}

	// test4
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 6; j++)
			data[j] = i * 6 + j;
		tmp.setTrans(true);
		matrices[i].copyFromHost(tmp, true);
	}
	NVMatrix colTrue;
	assembleNVMatrix(matrices, colTrue, 1);
	colTrue.printShape("colTrue");
	colTrue.print(2,12);

	for (int i = 0; i < 4; i++) {
		matrices[i].resize(0,0);
	}
	splitNVMatrix(matrices, colTrue, 1);
	for (int i = 0; i < 4; i++) {
		sprintf(a, "colTrue%d", i);
		matrices[i].printShape(a);
		matrices[i].print(2,3);
	}

	printf("finished testAssembleMatrix()\n");
}

void testAssembleMatrix1() {
	printf("start testAssembleMatrix1()\n");
	vector<NVMatrix> matrices(4);
	NVMatrix mat(32*32*3, 60000);
	mat.setTrans(false);
	NVReadFromFile(mat, "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin");
	splitNVMatrix(matrices, mat, 0);
	for (int i = 0; i < 4; i++) {
		char a[100];
		sprintf(a, "/scratch0/qwang37/cifar-10-batches-bin/testAssemble/r%d.bin", i);
		NVSaveToFile(matrices[i], a);
	}
	NVMatrix ass;
	assembleNVMatrix(matrices, ass, 0);
	NVSaveToFile(ass, "/scratch0/qwang37/cifar-10-batches-bin/testAssemble.bin");
	printf("completed testAssembleMatrix1()!\n");
}

void testAssembleMatrix2() {
	printf("start testAssembleMatrix2()\n");
	vector<NVMatrix> sub(2);
	NVMatrix mat(32*32*3, 60000);
	mat.setTrans(false);
	NVReadFromFile(mat, "/scratch0/qwang37/cifar-10-batches-bin/cifar_raw.bin");
	splitNVMatrix(sub[0], sub[1], mat, 1000, 2072, 0);
	for (int i = 0; i < 2; i++) {
		char a[100];
		sprintf(a, "/scratch0/qwang37/cifar-10-batches-bin/testAssemble/r%d.bin", i);
		NVSaveToFile(sub[i], a);
	}
	NVMatrix ass;
	assembleNVMatrix(sub[0], sub[1], ass, 0);
	printf("finished assembleNVMatrix!\n");
	NVSaveToFile(ass, "/scratch0/qwang37/cifar-10-batches-bin/rowAssemble.bin");

	splitNVMatrix(sub[0], sub[1], mat, 20000, 40000, 1);
	for (int i = 0; i < 2; i++) {
		char a[100];
		sprintf(a, "/scratch0/qwang37/cifar-10-batches-bin/testAssemble/c%d.bin", i);
		NVSaveToFile(sub[i], a);
	}
	NVMatrix ass1;
	assembleNVMatrix(sub[0], sub[1], ass1, 1);
	NVSaveToFile(ass1, "/scratch0/qwang37/cifar-10-batches-bin/colAssemble.bin");
	printf("completed testAssembleMatrix2()!\n");
}

void testGenFilterMask() {
	curandState* devStates = init_cuda_rand(100);
	printf("start testGenFilterMask()!\n");
	NVMatrix a;
	genFilterMask(a, 10, 10, 0.2, devStates);
	a.printShape("a");
	a.print(10,10);
}

void testAbs() {
	NVMatrix a;
	a.resize(16*10,10);
	a.setTrans(false);
	a.randomizeGaussian();
	NVMatrix b;
	NVMask maxMask;
	convCompeteAbs(a, b, maxMask, 10, 2, 2, 4, 10);
	b.print(16*10, 10);
}
