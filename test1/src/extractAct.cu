#include <vector>
#include <iostream>

#include <nvmatrix.cuh>
#include "routines.cuh"
#include "layer_kernels.cuh"
#include <cudaconv2.cuh>
using namespace std;

extern LayerOpt opt1, opt2, opt3, opt4, optTop;
extern FILE* pFile;

void extractAct(string actPath) {
	////assignOpt();
	printf("starting extractAct()!\n");
	fprintf(pFile, "starting extractAct()!\n");

	vector<FILE*> pActFiles(opt1.labelSize);
	for (int i = 0; i < opt1.labelSize; i++)
		pActFiles[i] = fopen((actPath+"/act_class"+char(i+'0')).c_str(), "wt");

	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cublasInit();

	// data and parameters storage
	NVMatrix act0;
	NVMatrix act1, act1Pool, act1PoolMax;
	NVMatrix act2, act2Pool, act2PoolMax;
	NVMatrix act3, act3Pool, act3PoolMax;
	NVMatrix act4, act4Max;
	NVMatrix actTop;
	NVMatrix act1Grad, act1PoolGrad, act1PoolMaxGrad;
	NVMatrix act2Grad, act2PoolGrad, act2PoolMaxGrad;
	NVMatrix act3Grad, act3PoolGrad, act3PoolMaxGrad;
	NVMatrix act4Grad, act4MaxGrad;
	NVMatrix actTopGrad;

	NVMatrix weight1, weight2, weight3, weight4, weightTop;
	NVMatrix bias1, bias2, bias3, bias4, biasTop; // bias4 is just an all-zero dummy vector
	// initialize parameters

	// initialize parameters
	weight1.resize(opt1.numVis, opt1.numFilters);
	weight2.resize(opt2.numVis, opt2.numFilters);
	if (strcmp(opt3.layerType, "local") == 0)
		weight3.resize(opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters);
	else if (strcmp(opt3.layerType, "conv") == 0)
		weight3.resize(opt3.numVis, opt3.numFilters);
	weight4.resize(opt4.numVis, opt4.numFilters);
	weightTop.resize(optTop.numVis, optTop.numFilters);

	bias1.resize(opt1.numFilters, 1);
	bias2.resize(opt2.numFilters, 1);
	if (strcmp(opt3.layerType, "local") == 0)
		bias3.resize(opt3.numFilters * opt3.outX * opt3.outX, 1);
	else if (strcmp(opt3.layerType, "conv") == 0)
		bias3.resize(opt3.numFilters, 1);
	bias4.resize(1, opt4.numFilters);
	bias4.setTrans(true);
	biasTop.resize(1, optTop.numFilters);
	biasTop.setTrans(true);

	NVReadFromFile(weight1, opt1.weightPath + "/weight1.bin");
	NVReadFromFile(weight2, opt1.weightPath + "/weight2.bin");
	NVReadFromFile(weight3, opt1.weightPath + "/weight3.bin");
	NVReadFromFile(weight4, opt1.weightPath + "/weight4.bin");
	NVReadFromFile(weightTop, opt1.weightPath + "/weightTop.bin");

	NVReadFromFile(bias1, opt1.weightPath + "/bias1.bin");
	NVReadFromFile(bias2, opt1.weightPath + "/bias2.bin");
	NVReadFromFile(bias3, opt1.weightPath + "/bias3.bin");
	NVReadFromFile(bias4, opt1.weightPath + "/bias4.bin");
	NVReadFromFile(biasTop, opt1.weightPath + "/biasTop.bin");

	// read data to host memory (and labels to the GPU memory)
	int imPixels = 32*32*opt1.numChannels;
	int batchSize = opt1.batchSize;
	int testBatchNum = opt1.numTest / batchSize;
	vector<Matrix*> CPUTest(testBatchNum);
	vector<NVMatrix*> GPUTest(testBatchNum*opt1.numViews);
	vector<NVMatrix*> GPURawLabelTest(testBatchNum);
	vector<Matrix*> CPURawLabelTest(testBatchNum);

	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		GPUTest[batch] = new NVMatrix();
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);
		CPURawLabelTest[batch] = new Matrix(1, batchSize);
		CPURawLabelTest[batch]->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+batch*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest[batch], (opt1.dataPath + "/cifar_labels.bin").c_str(), opt1.numTrain+batch*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar100_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar100_fine_labels.bin", opt1.numTrain+batch*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest[batch], (opt1.dataPath + "/cifar100_fine_labels.bin").c_str(), opt1.numTrain+batch*batchSize);
		}
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		GPUTest.push_back(new NVMatrix());
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);
		CPURawLabelTest.push_back(new Matrix(1, batchSize));
		CPURawLabelTest.back()->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest.back(), (opt1.dataPath + "/cifar_labels.bin").c_str(), opt1.numTrain+testBatchNum*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar100_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar100_fine_labels.bin", opt1.numTrain+testBatchNum*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest.back(), (opt1.dataPath + "/cifar100_fine_labels.bin").c_str(), opt1.numTrain+testBatchNum*batchSize);
		}
	}

	//multiViewDataProvider(CPUTest, GPUTest, opt1, opt1.numViews, opt1.whitened); // copy data to the GPU side
	cropDataProvider(CPUTest, GPUTest, opt1, opt1.numViews, opt1.whitened);

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

	NVMask maxMask1, maxMask2, maxMask3, maxMask4;


	for (int batch = 0; batch < CPUTest.size(); batch++) {
		batchSize = CPUTest[batch]->getNumCols();
			// ====forward pass====
			// 0->1
//			cout << "0->1\n";
			//original
			GPUTest[batch]->copy(act0);
			act0.scale(opt1.inputScale);
			activateConv(act0, act1, weight1, bias1, opt1);
			act1Pool.transpose(false);
			convLocalPool(act1, act1Pool, opt1.numFilters, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX, MaxPooler());
			convCompeteOut(act1Pool, act1PoolMax, maxMask1, opt1.numFilters, opt1.maxOutPoolSize, opt1.maxOutPoolStride, opt1.poolOutX, batchSize);

			// 1->2
//			cout << "1->2\n";
			//original
			activateConv(act1PoolMax, act2, weight2, bias2, opt2);
			act2Pool.transpose(false);
			convLocalPool(act2, act2Pool, opt2.numFilters, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX, MaxPooler());
			convCompeteOut(act2Pool, act2PoolMax, maxMask2, opt2.numFilters, opt2.maxOutPoolSize, opt2.maxOutPoolStride, opt2.poolOutX, batchSize);


			// 2->3
//			cout << "2->3\n";
			// original
			if (strcmp(opt3.layerType, "local") == 0)
				activateLocal(act2PoolMax, act3, weight3, bias3, opt3);
			else if (strcmp(opt3.layerType, "conv") == 0)
				activateConv(act2PoolMax, act3, weight3, bias3, opt3);
			act3Pool.transpose(false);
			convLocalPool(act3, act3Pool, opt3.numFilters, opt3.poolSize, opt3.poolStartX, opt3.poolStride, opt3.poolOutX, MaxPooler());
			convCompeteOut(act3Pool, act3PoolMax, maxMask3, opt3.numFilters, opt3.maxOutPoolSize, opt3.maxOutPoolStride, opt3.poolOutX, batchSize);


			// 3->4
//			cout << "3->4\n";
			// original

			act3PoolMax.transpose(true);
			activate(act3PoolMax, act4, weight4, bias4, 0, 1);
			act3PoolMax.transpose(false);
			act4.transpose(false);
			convCompeteOut(act4, act4Max, maxMask4, opt4.numFilters/opt4.outX/opt4.outX, opt4.maxOutPoolSize, opt4.maxOutPoolStride, opt4.outX, batchSize);


			// 4->top
//			cout << "4->top\n";
			act4Max.transpose(true);
			activate(act4Max, actTop, weightTop, biasTop, 0, 1);
			act4Max.transpose(false);

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
			cost += correctProbs.sum();
			cost1 += trueLabelLogProbs.sum();

			// extract the activations
			int* hmMaxMask1 = maxMask1.copyToHost();
			int* hmMaxMask2 = maxMask2.copyToHost();
			int* hmMaxMask3 = maxMask3.copyToHost();
			int* hmMaxMask4 = maxMask4.copyToHost();


			//CPURawLabelTest[batch]->print();
			//cout << "here\n";
			//cout << CPURawLabelTest[batch]->getCell(0,1) << "\n";

			for (int s = 0; s < batchSize; s++) {
				int lab = CPURawLabelTest[batch]->getCell(0,s);
				//cout << 'c' << lab;
				//cout << hmMaxMask4[0];

				for (int l = s; l < maxMask4.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask4[l]);
				for (int l = s; l < maxMask3.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask3[l]);
				for (int l = s; l < maxMask2.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask2[l]);
				for (int l = s; l < maxMask1.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask1[l]);
				fprintf(pActFiles[lab], "\n");


			}



	}//for (batches)

	for (int i = 0; i < pActFiles.size(); i++)
		fclose(pActFiles[i]);

	cudaThreadSynchronize();
	cost /= opt1.numTest;
	cost1 /= opt1.numTest;
	printf("\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
			(float)(clock() - startClock)/CLOCKS_PER_SEC);
	printf("extractAct() complete!\n");

	fprintf(pFile, "\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
			(float)(clock() - startClock)/CLOCKS_PER_SEC);
	fprintf(pFile, "extractAct() complete!\n");
}

void extractActMaxout(string actPath) {
	////assignOpt();
	printf("starting extractActMaxout()!\n");
	fprintf(pFile, "starting extractActMaxout()!\n");

	vector<FILE*> pActFiles(opt1.labelSize);
	for (int i = 0; i < opt1.labelSize; i++)
		pActFiles[i] = fopen((actPath+"/act_class"+char(i+'0')).c_str(), "wt");

	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cublasInit();

	// data and parameters storage
	NVMatrix act0;
	NVMatrix act1, act1Pool, act1PoolMax;
	NVMatrix act2, act2Pool, act2PoolMax;
	NVMatrix act3, act3Pool, act3PoolMax;
	NVMatrix act4, act4Max;
	NVMatrix actTop;
	NVMatrix act1Grad, act1PoolGrad, act1PoolMaxGrad;
	NVMatrix act2Grad, act2PoolGrad, act2PoolMaxGrad;
	NVMatrix act3Grad, act3PoolGrad, act3PoolMaxGrad;
	NVMatrix act4Grad, act4MaxGrad;
	NVMatrix actTopGrad;

	NVMatrix weight1, weight2, weight3, weight4, weightTop;
	NVMatrix bias1, bias2, bias3, bias4, biasTop; // bias4 is just an all-zero dummy vector
	// initialize parameters

	// initialize parameters
	weight1.resize(opt1.numVis, opt1.numFilters);
	weight2.resize(opt2.numVis, opt2.numFilters);
	if (strcmp(opt3.layerType, "local") == 0)
		weight3.resize(opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters);
	else if (strcmp(opt3.layerType, "conv") == 0)
		weight3.resize(opt3.numVis, opt3.numFilters);
	weight4.resize(opt4.numVis, opt4.numFilters);
	weightTop.resize(optTop.numVis, optTop.numFilters);

	bias1.resize(opt1.numFilters, 1);
	bias2.resize(opt2.numFilters, 1);
	if (strcmp(opt3.layerType, "local") == 0)
		bias3.resize(opt3.numFilters * opt3.outX * opt3.outX, 1);
	else if (strcmp(opt3.layerType, "conv") == 0)
		bias3.resize(opt3.numFilters, 1);
	bias4.resize(1, opt4.numFilters);
	bias4.setTrans(true);
	biasTop.resize(1, optTop.numFilters);
	biasTop.setTrans(true);

	NVReadFromFile(weight1, opt1.weightPath + "/weight1.bin");
	NVReadFromFile(weight2, opt1.weightPath + "/weight2.bin");
	NVReadFromFile(weight3, opt1.weightPath + "/weight3.bin");
	NVReadFromFile(weight4, opt1.weightPath + "/weight4.bin");
	NVReadFromFile(weightTop, opt1.weightPath + "/weightTop.bin");

	NVReadFromFile(bias1, opt1.weightPath + "/bias1.bin");
	NVReadFromFile(bias2, opt1.weightPath + "/bias2.bin");
	NVReadFromFile(bias3, opt1.weightPath + "/bias3.bin");
	NVReadFromFile(bias4, opt1.weightPath + "/bias4.bin");
	NVReadFromFile(biasTop, opt1.weightPath + "/biasTop.bin");

	// read data to host memory (and labels to the GPU memory)
	int imPixels = 32*32*opt1.numChannels;
	int batchSize = opt1.batchSize;
	int testBatchNum = opt1.numTest / batchSize;
	vector<Matrix*> CPUTest(testBatchNum);
	vector<NVMatrix*> GPUTest(testBatchNum*opt1.numViews);
	vector<NVMatrix*> GPURawLabelTest(testBatchNum);
	vector<Matrix*> CPURawLabelTest(testBatchNum);

	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		GPUTest[batch] = new NVMatrix();
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);
		CPURawLabelTest[batch] = new Matrix(1, batchSize);
		CPURawLabelTest[batch]->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+batch*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest[batch], (opt1.dataPath + "/cifar_labels.bin").c_str(), opt1.numTrain+batch*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar100_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar100_fine_labels.bin", opt1.numTrain+batch*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest[batch], (opt1.dataPath + "/cifar100_fine_labels.bin").c_str(), opt1.numTrain+batch*batchSize);
		}
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		GPUTest.push_back(new NVMatrix());
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);
		CPURawLabelTest.push_back(new Matrix(1, batchSize));
		CPURawLabelTest.back()->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest.back(), (opt1.dataPath + "/cifar_labels.bin").c_str(), opt1.numTrain+testBatchNum*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar100_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar100_fine_labels.bin", opt1.numTrain+testBatchNum*batchSize);
			hmRawLabelReadFromFile(*CPURawLabelTest.back(), (opt1.dataPath + "/cifar100_fine_labels.bin").c_str(), opt1.numTrain+testBatchNum*batchSize);
		}
	}

	//multiViewDataProvider(CPUTest, GPUTest, opt1, opt1.numViews, opt1.whitened); // copy data to the GPU side
	cropDataProvider(CPUTest, GPUTest, opt1, opt1.numViews, opt1.whitened);

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

	NVMask maxMask1, maxMask2, maxMask3, maxMask4;


	for (int batch = 0; batch < CPUTest.size(); batch++) {
		batchSize = CPUTest[batch]->getNumCols();
			// ====forward pass====
			// 0->1
//			cout << "0->1\n";
			//original
			GPUTest[batch]->copy(act0);
			act0.scale(opt1.inputScale);
			activateConv(act0, act1, weight1, bias1, opt1);
			act1Pool.transpose(false);
			convLocalPool(act1, act1Pool, opt1.numFilters, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX, MaxPooler());
			convMaxOut(act1Pool, act1PoolMax, maxMask1, opt1.numFilters, opt1.maxOutPoolSize, opt1.maxOutPoolStride, opt1.poolOutX, batchSize);

			// 1->2
//			cout << "1->2\n";
			//original
			activateConv(act1PoolMax, act2, weight2, bias2, opt2);
			act2Pool.transpose(false);
			convLocalPool(act2, act2Pool, opt2.numFilters, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX, MaxPooler());
			convMaxOut(act2Pool, act2PoolMax, maxMask2, opt2.numFilters, opt2.maxOutPoolSize, opt2.maxOutPoolStride, opt2.poolOutX, batchSize);


			// 2->3
//			cout << "2->3\n";
			// original
			if (strcmp(opt3.layerType, "local") == 0)
				activateLocal(act2PoolMax, act3, weight3, bias3, opt3);
			else if (strcmp(opt3.layerType, "conv") == 0)
				activateConv(act2PoolMax, act3, weight3, bias3, opt3);
			act3Pool.transpose(false);
			convLocalPool(act3, act3Pool, opt3.numFilters, opt3.poolSize, opt3.poolStartX, opt3.poolStride, opt3.poolOutX, MaxPooler());
			convMaxOut(act3Pool, act3PoolMax, maxMask3, opt3.numFilters, opt3.maxOutPoolSize, opt3.maxOutPoolStride, opt3.poolOutX, batchSize);


			// 3->4
//			cout << "3->4\n";
			// original

			act3PoolMax.transpose(true);
			activate(act3PoolMax, act4, weight4, bias4, 0, 1);
			act3PoolMax.transpose(false);
			act4.transpose(false);
			convMaxOut(act4, act4Max, maxMask4, opt4.numFilters/opt4.outX/opt4.outX, opt4.maxOutPoolSize, opt4.maxOutPoolStride, opt4.outX, batchSize);


			// 4->top
//			cout << "4->top\n";
			act4Max.transpose(true);
			activate(act4Max, actTop, weightTop, biasTop, 0, 1);
			act4Max.transpose(false);

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
			cost += correctProbs.sum();
			cost1 += trueLabelLogProbs.sum();

			// extract the activations
			int* hmMaxMask1 = maxMask1.copyToHost();
			int* hmMaxMask2 = maxMask2.copyToHost();
			int* hmMaxMask3 = maxMask3.copyToHost();
			int* hmMaxMask4 = maxMask4.copyToHost();


			//CPURawLabelTest[batch]->print();
			//cout << "here\n";
			//cout << CPURawLabelTest[batch]->getCell(0,1) << "\n";

			for (int s = 0; s < batchSize; s++) {
				int lab = CPURawLabelTest[batch]->getCell(0,s);
				//cout << 'c' << lab;
				//cout << hmMaxMask4[0];

				for (int l = s; l < maxMask4.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask4[l]);
				for (int l = s; l < maxMask3.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask3[l]);
				for (int l = s; l < maxMask2.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask2[l]);
				for (int l = s; l < maxMask1.getSize(); l += batchSize)
					fprintf(pActFiles[lab], "%d", hmMaxMask1[l]);
				fprintf(pActFiles[lab], "\n");


			}



	}//for (batches)

	for (int i = 0; i < pActFiles.size(); i++)
		fclose(pActFiles[i]);

	cudaThreadSynchronize();
	cost /= opt1.numTest;
	cost1 /= opt1.numTest;
	printf("\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
			(float)(clock() - startClock)/CLOCKS_PER_SEC);
	printf("extractActMaxout() complete!\n");

	fprintf(pFile, "\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
			(float)(clock() - startClock)/CLOCKS_PER_SEC);
	fprintf(pFile, "extractActMaxout() complete!\n");
}
