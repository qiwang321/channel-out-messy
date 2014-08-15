#include <vector>
#include <iostream>
#include <string>

#include "routines.cuh"
#include "layer_kernels.cuh"
#include <cudaconv2.cuh>

using namespace std;

extern LayerOpt opt1, opt2, opt3, opt4, optTop;
extern FILE* pFile;
extern float perturbScale;
//extern float keepProb;
//extern float widerScale;

void finetune_rnorm_competeOut_maxtop() {
	////assignOpt();
	printf("starting finetune_rnorm_competeOut_maxtop()!\nkeepProb = %f-%f-%f\nperturbScale = %f\n", opt1.keepStartRate, opt1.keepIncRate, opt1.keepEndRate, perturbScale);
	printf("inputKeepProb = %f-%f-%f\n", opt1.keepInputStartRate, opt1.keepInputIncRate, opt1.keepInputEndRate);
	printf("image sizes\nlayer1: %d\nlayer2: %d\nlayer3: %d\nlayer4: %d\n", opt1.imSize, opt2.imSize, opt3.imSize, opt4.imSize);
	printf("image channels\nlayer1: %d\nlayer2: %d\nlayer3: %d\nlayer4: %d\nlayerTop: %d\n", opt1.numChannels, opt2.numChannels, opt3.numChannels, opt4.numChannels, opt4.numFilters);
	printf("learning rates: layer1--%f, layer2--%f, layer3--%f, layer4--%f, layerTop--%f\n", opt1.lrW, opt2.lrW, opt3.lrW, opt4.lrW, optTop.lrW);
	printf("bias rates: layer1--%f, layer2--%f, layer3--%f, layer4--%f, layerTop--%f\n", opt1.lrB, opt2.lrB, opt3.lrB, opt4.lrB, optTop.lrB);
	printf("inputScale--%f\n", opt1.inputScale);
	printf("lrStartRate = %f, momStartRate = %f\n", opt1.lrStartScale, opt1.momStartScale);
	printf("lrEndRate = %f, momEndRate = %f\n", opt1.lrMinRate, opt1.momMaxRate);
	printf("lrDecayFactor = %f, momIncFactor = %f\n", opt1.lrDecayFactor, opt1.momIncFactor);
	printf("lr schedule = %s, mom schedule = %s\n", opt1.lrDecayType, opt1.momIncType);
	printf("competeMax sizes: layer1 -- %d, layer2 -- %d, layer3 -- %d, layer4 -- %d\n", opt1.maxOutPoolSize, opt2.maxOutPoolSize, opt3.maxOutPoolSize, opt4.maxOutPoolSize);
	printf("competeMax strides: layer1 -- %d, layer2 -- %d, layer3 -- %d, layer4 -- %d\n", opt1.maxOutPoolStride, opt2.maxOutPoolStride, opt3.maxOutPoolStride, opt4.maxOutPoolStride);
	printf("labelSize = %d\n", opt1.labelSize);
	printf("flip = %d\n", opt1.flip);


	fprintf(pFile, "starting finetune_rnorm_competeOut_maxtop()!\nkeepProb = %f-%f-%f\nperturbScale = %f\n", opt1.keepStartRate, opt1.keepIncRate, opt1.keepEndRate, perturbScale);
	fprintf(pFile, "inputKeepProb = %f-%f-%f\n", opt1.keepInputStartRate, opt1.keepInputIncRate, opt1.keepInputEndRate);
	fprintf(pFile, "image sizes\nlayer1: %d\nlayer2: %d\nlayer3: %d\nlayer4: %d\n", opt1.imSize, opt2.imSize, opt3.imSize, opt4.imSize);
	fprintf(pFile, "image channels\nlayer1: %d\nlayer2: %d\nlayer3: %d\nlayer4: %d\nlayerTop: %d\n", opt1.numChannels, opt2.numChannels, opt3.numChannels, opt4.numChannels, opt4.numFilters);
	fprintf(pFile, "learning rates: layer1--%f, layer2--%f, layer3--%f, layer4--%f, layerTop--%f\n", opt1.lrW, opt2.lrW, opt3.lrW, opt4.lrW, optTop.lrW);
	fprintf(pFile, "bias rates: layer1--%f, layer2--%f, layer3--%f, layer4--%f, layerTop--%f\n", opt1.lrB, opt2.lrB, opt3.lrB, opt4.lrB, optTop.lrB);
	fprintf(pFile, "inputScale--%f\n", opt1.inputScale);
	fprintf(pFile, "lrStartRate = %f, momStartRate = %f\n", opt1.lrStartScale, opt1.momStartScale);
	fprintf(pFile, "lrEndRate = %f, momEndRate = %f\n", opt1.lrMinRate, opt1.momMaxRate);
	fprintf(pFile, "lrDecayFactor = %f, momIncFactor = %f\n", opt1.lrDecayFactor, opt1.momIncFactor);
	fprintf(pFile, "lr schedule = %s, mom schedule = %s\n", opt1.lrDecayType, opt1.momIncType);
	fprintf(pFile, "competeMax sizes: layer1 -- %d, layer2 -- %d, layer3 -- %d, layer4 -- %d\n", opt1.maxOutPoolSize, opt2.maxOutPoolSize, opt3.maxOutPoolSize, opt4.maxOutPoolSize);
	fprintf(pFile, "competeMax strides: layer1 -- %d, layer2 -- %d, layer3 -- %d, layer4 -- %d\n", opt1.maxOutPoolStride, opt2.maxOutPoolStride, opt3.maxOutPoolStride, opt4.maxOutPoolStride);
	fprintf(pFile, "labelSize = %d\n", opt1.labelSize);
	fprintf(pFile, "flip = %d\n", opt1.flip);

	// data and parameters storage
	NVMatrix act0;
	NVMatrix act1, act1Pool, act1PoolMax;
	NVMatrix act2, act2Pool, act2PoolMax;
	NVMatrix act3, act3Pool, act3PoolMax;
	NVMatrix act4, act4Max;
	NVMatrix actTop, actTopMax;
	NVMatrix act1Grad, act1PoolGrad, act1PoolMaxGrad;
	NVMatrix act2Grad, act2PoolGrad, act2PoolMaxGrad;
	NVMatrix act3Grad, act3PoolGrad, act3PoolMaxGrad;
	NVMatrix act4Grad, act4MaxGrad;
	NVMatrix actTopGrad, actTopMaxGrad;


	NVMatrix weight1, weight2, weight3, weight4, weightTop;
	NVMatrix weight1Grad, weight2Grad, weight3Grad, weight4Grad, weightTopGrad;
	NVMatrix weight1Inc, weight2Inc, weight3Inc, weight4Inc, weightTopInc;
	NVMatrix weight1GradTmp, weight2GradTmp, weight3GradTmp, weight4GradTmp, weightTopGradTmp;

	NVMatrix bias1, bias2, bias3, bias4, biasTop; // bias4 is just an all-zero dummy vector
	NVMatrix bias1Grad, bias2Grad, bias3Grad, bias4Grad, biasTopGrad;
	NVMatrix bias1Inc, bias2Inc, bias3Inc, bias4Inc, biasTopInc;

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

	if (opt1.loadParam) {
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

	}
	else {
		weight1.randomizeUniform(); weight1.scale(2*opt1.initstv); weight1.addScalar(-opt1.initstv);
		weight2.randomizeUniform(); weight2.scale(2*opt2.initstv); weight2.addScalar(-opt2.initstv);
		weight3.randomizeUniform(); weight3.scale(2*opt3.initstv); weight3.addScalar(-opt3.initstv);
		weight4.randomizeUniform(); weight4.scale(2*opt4.initstv); weight4.addScalar(-opt4.initstv);
		weightTop.randomizeUniform(); weightTop.scale(2*optTop.initstv); weightTop.addScalar(-optTop.initstv);

		bias1.apply(NVMatrixOps::Zero());
		bias2.apply(NVMatrixOps::Zero());
		bias3.apply(NVMatrixOps::Zero());
		bias4.apply(NVMatrixOps::Zero());
		biasTop.apply(NVMatrixOps::Zero());
	}


	initWeights(weight1Inc, opt1.numVis, opt1.numFilters, false, 0.0); initWeights(weight1Grad, opt1.numVis, opt1.numFilters, false, 0.0);
	initWeights(weight2Inc, opt2.numVis, opt2.numFilters, false, 0.0); initWeights(weight2Grad, opt2.numVis, opt2.numFilters, false, 0.0);
	if (strcmp(opt3.layerType, "local") == 0) {
		initWeights(weight3Inc, opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters, false, 0.0);
		initWeights(weight3Grad, opt3.numVis * opt3.outX * opt3.outX, opt3.numFilters, false, 0.0);
	}
	else if (strcmp(opt3.layerType, "conv") == 0) {
		initWeights(weight3Inc, opt3.numVis, opt3.numFilters, false, 0.0);
		initWeights(weight3Grad, opt3.numVis, opt3.numFilters, false, 0.0);
	}
	initWeights(weight4Inc, opt4.numVis, opt4.numFilters, false, 0.0); initWeights(weight4Grad, opt4.numVis, opt4.numFilters, false, 0.0);
	initWeights(weightTopInc, optTop.numVis, optTop.numFilters, false, 0.0); initWeights(weightTopGrad, optTop.numVis, optTop.numFilters, false, 0.0);


	initWeights(bias1Inc, opt1.numFilters, 1, false, 0.0); initWeights(bias1Grad, opt1.numFilters, 1, false, 0.0);
	initWeights(bias2Inc, opt2.numFilters, 1, false, 0.0); initWeights(bias2Grad, opt2.numFilters, 1, false, 0.0);
	if (strcmp(opt3.layerType, "local") == 0) {
		initWeights(bias3Inc, opt3.numFilters * opt3.outX * opt3.outX, 1, false, 0.0);
		initWeights(bias3Grad, opt3.numFilters * opt3.outX * opt3.outX, 1, false, 0.0);
	}
	else if (strcmp(opt3.layerType, "conv") == 0) {
		initWeights(bias3Inc, opt3.numFilters, 1, false, 0.0);
		initWeights(bias3Grad, opt3.numFilters, 1, false, 0.0);
	}
	initWeights(bias4Inc, 1, opt4.numFilters, false, 0.0); initWeights(bias4Grad, 1, opt4.numFilters, false, 0.0);
	initWeights(biasTopInc, 1, optTop.numFilters, true, 0.0); initWeights(biasTopGrad, 1, optTop.numFilters, true, 0.0);


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
		GPURawLabelTrain[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTrain[batch]->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTrain[batch], opt1.dataPath + "/cifar_whitened.bin", batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTrain[batch], opt1.dataPath + "/cifar_labels.bin", batch*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTrain[batch], opt1.dataPath + "/cifar100_whitened.bin", batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTrain[batch], opt1.dataPath + "/cifar100_fine_labels.bin", batch*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "stl10") == 0) {
			hmReadFromFile(*CPUTrain[batch], opt1.dataPath + "/stl10_32x32_whitened.bin", batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTrain[batch], opt1.dataPath + "/stl10_32x32_labels.bin", batch*batchSize);
		}
	}
	batchSize = opt1.numTrain % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTrain.push_back(new Matrix(imPixels, batchSize));
		CPUTrain.back()->setTrans(false);
		GPUTrain.push_back(new NVMatrix());
		GPURawLabelTrain.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTrain.back()->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTrain.back(), opt1.dataPath + "/cifar_whitened.bin", trainBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTrain.back(), opt1.dataPath + "/cifar_labels.bin", trainBatchNum*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTrain.back(), opt1.dataPath + "/cifar100_whitened.bin", trainBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTrain.back(), opt1.dataPath + "/cifar100_fine_labels.bin", trainBatchNum*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "stl10") == 0) {
			hmReadFromFile(*CPUTrain.back(), opt1.dataPath + "/stl10_32x32_whitened.bin", trainBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTrain.back(), opt1.dataPath + "/stl10_32x32_labels.bin", trainBatchNum*batchSize);
		}
	}
	// test set
	batchSize = opt1.batchSize;
	for (int batch = 0; batch < testBatchNum; batch++) {
		CPUTest[batch] = new Matrix(imPixels, batchSize);
		CPUTest[batch]->setTrans(false);
		GPUTest[batch] = new NVMatrix();
		GPURawLabelTest[batch] = new NVMatrix(1, batchSize);
		GPURawLabelTest[batch]->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+batch*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/cifar100_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/cifar100_fine_labels.bin", opt1.numTrain+batch*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "stl10") == 0) {
			hmReadFromFile(*CPUTest[batch], opt1.dataPath + "/stl10_32x32_whitened.bin", opt1.numTrain+batch*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest[batch], opt1.dataPath + "/stl10_32x32_labels.bin", opt1.numTrain+batch*batchSize);
		}
	}
	batchSize = opt1.numTest % opt1.batchSize; // the last batch
	if (batchSize > 0) {
		CPUTest.push_back(new Matrix(imPixels, batchSize));
		CPUTest.back()->setTrans(false);
		GPUTest.push_back(new NVMatrix());
		GPURawLabelTest.push_back(new NVMatrix(1, batchSize));
		GPURawLabelTest.back()->setTrans(false);

		if (strcmp(opt1.exp.c_str(), "cifar10") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar_labels.bin", opt1.numTrain+testBatchNum*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "cifar100") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/cifar100_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/cifar100_fine_labels.bin", opt1.numTrain+testBatchNum*batchSize);
		}
		else if (strcmp(opt1.exp.c_str(), "stl10") == 0) {
			hmReadFromFile(*CPUTest.back(), opt1.dataPath + "/stl10_32x32_whitened.bin", opt1.numTrain+testBatchNum*batchSize);
			NVRawLabelReadFromFile(*GPURawLabelTest.back(), opt1.dataPath + "/stl10_32x32_labels.bin", opt1.numTrain+testBatchNum*batchSize);
		}
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

	NVMask maxMask1, maxMask2, maxMask3, maxMask4, maxMaskTop;
	NVMatrix dropMask0, dropMask1, dropMask2, dropMask3, dropMask4;
	NVMatrix widerMask1, widerMask2, widerMask3, widerMask4;

	// normalization
	NVMatrix normCol1, normCol2, normCol3, normCol4, normColTop;
	NVMatrix tmp1, tmp2, tmp3, tmp4, tmpTop;
	float lr_scale = opt1.lrStartScale, mom_scale = opt1.momStartScale;
	float keepProb = opt1.keepStartRate;
	float keepInputProb = opt1.keepInputStartRate;

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
//			cout << "0->1\n";
			//original
			GPUTrain[batch]->copy(act0);
			dropMask0.resize(act0);
			dropMask0.randomizeBinary(keepInputProb); // input part
			act0.eltwiseMult(dropMask0);
			act0.scale(opt1.inputScale);
			act0.scale(1.0 / keepInputProb); // very, very important!
			activateConv(act0, act1, weight1, bias1, opt1);
			act1Pool.transpose(false);
			convLocalPool(act1, act1Pool, opt1.numFilters, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX, MaxPooler());
			convCompeteOut(act1Pool, act1PoolMax, maxMask1, opt1.numFilters, opt1.maxOutPoolSize, opt1.maxOutPoolStride, opt1.poolOutX, batchSize);

			// 1->2
//			cout << "1->2\n";
			//original
			dropMask1.resize(act1PoolMax);
			dropMask1.randomizeBinary(keepProb);
			act1PoolMax.eltwiseMult(dropMask1);
			act1PoolMax.scale(1.0 / keepProb); // very, very important!


			activateConv(act1PoolMax, act2, weight2, bias2, opt2);
			act2Pool.transpose(false);
			convLocalPool(act2, act2Pool, opt2.numFilters, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX, MaxPooler());
			convCompeteOut(act2Pool, act2PoolMax, maxMask2, opt2.numFilters, opt2.maxOutPoolSize, opt2.maxOutPoolStride, opt2.poolOutX, batchSize);


			// 2->3
//			cout << "2->3\n";
			// original
			dropMask2.resize(act2PoolMax);
			dropMask2.randomizeBinary(keepProb);
			act2PoolMax.eltwiseMult(dropMask2);
			act2PoolMax.scale(1.0 / keepProb);

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
			dropMask3.resize(act3PoolMax);
			dropMask3.randomizeBinary(keepProb);
			act3PoolMax.eltwiseMult(dropMask3);
			act3PoolMax.scale(1.0 / keepProb);

			act3PoolMax.transpose(true);
			activate(act3PoolMax, act4, weight4, bias4, 0, 1);
			act3PoolMax.transpose(false);
			act4.transpose(false);
			convCompeteOut(act4, act4Max, maxMask4, opt4.numFilters/opt4.outX/opt4.outX, opt4.maxOutPoolSize, opt4.maxOutPoolStride, opt4.outX, batchSize);


			// 4->top
//			cout << "4->top\n";
			dropMask4.resize(act4Max);
			dropMask4.randomizeBinary(keepProb);
			act4Max.eltwiseMult(dropMask4);
			act4Max.scale(1.0 / keepProb);

			act4Max.transpose(true);
			activate(act4Max, actTop, weightTop, biasTop, 0, 1);
			act4Max.transpose(false);
			actTop.transpose(false);
			convMaxOut(actTop, actTopMax, maxMaskTop, optTop.numFilters, optTop.maxOutPoolSize, optTop.maxOutPoolStride, 1, batchSize);
			actTopMax.transpose(true);


			//softmax layer
			NVMatrix& max = actTopMax.max(1);
			actTopMax.addVector(max, -1);
			actTopMax.apply(NVMatrixOps::Exp());
			NVMatrix& sum = actTopMax.sum(1);
			actTopMax.eltwiseDivideByVector(sum);
			delete &max;
			delete &sum;

			// compute cost
			//actTopMax.transpose(true);
			computeLogregSoftmaxGrad(*GPURawLabelTrain[batch], actTopMax, actTopMaxGrad, false, 1); // orientation: col-major
			actTopMax.transpose(false);
			computeLogregCost(*GPURawLabelTrain[batch], actTopMax, trueLabelLogProbs, correctProbs); //labelLogProbs:(1, numCases); correctProbs:(1, numCases); orientation row-major
			cost += correctProbs.sum();
			cost1 += trueLabelLogProbs.sum();


			// ====== back pass ======
			// top -> 4, 3, 2, 1
			actTopMaxGrad.transpose(false);
			convMaxOutUndo(actTopMaxGrad, actTopGrad, maxMaskTop, optTop.numFilters, optTop.maxOutPoolStride, 1, batchSize);

//			cout << "top -> 4, 3, 2, 1\n";
			// weight update
			act4Max.transpose(false); actTopGrad.transpose(true);
			weightTopGrad.addProduct(act4Max, actTopGrad, 0, 1);
			biasTopGrad.addSum(actTopGrad, 0, 0, 1);
			// bp
			weightTop.transpose(true);
			act4MaxGrad.transpose(true);
			act4MaxGrad.addProduct(actTopGrad, weightTop, 0, 1);
			weightTop.transpose(false);

			// 4->3
//			cout << "4->3\n";
			act4MaxGrad.transpose(false); // convert back to row-major
			act4Max.transpose(false);
			act4MaxGrad.eltwiseMult(dropMask4);
			act4MaxGrad.scale(1.0 / keepProb);

			convCompeteOutUndo(act4MaxGrad, act4Grad, maxMask4, opt4.numFilters/opt4.outX/opt4.outX, opt4.maxOutPoolStride, opt4.outX, batchSize);

			act3PoolMax.transpose(false); act4Grad.transpose(true);
			weight4Grad.addProduct(act3PoolMax, act4Grad, 0, 1);
			bias4Grad.addSum(act4Grad, 0, 0, 1);
			// bp
			weight4.transpose(true);
			act3PoolMaxGrad.transpose(true);
			act3PoolMaxGrad.addProduct(act4Grad, weight4, 0, 1);
			weight4.transpose(false);

			// 3->2
//			cout << "3->2\n";
			act3PoolMaxGrad.transpose(false);
			act3PoolMax.transpose(false);
			act3PoolMaxGrad.eltwiseMult(dropMask3);
			act3PoolMaxGrad.scale(1.0 / keepProb);

			convCompeteOutUndo(act3PoolMaxGrad, act3PoolGrad, maxMask3, opt3.numFilters, opt3.maxOutPoolStride, opt3.poolOutX, batchSize);
			convLocalMaxUndo(act3, act3PoolGrad, act3Pool, act3Grad, opt3.poolSize, opt3.poolStartX, opt3.poolStride, opt3.poolOutX);

			if (strcmp(opt3.layerType, "local") == 0) {
				localWeightActs(act2PoolMax, act3Grad, weight3Grad, opt3.imSize, opt3.outX, opt3.outX, opt3.patchSize, opt3.paddingStart, 1, opt3.numChannels, 1);
				bias3Grad.addSum(act3Grad, 1, 0, 1);

				localImgActs(act3Grad, weight3, act2PoolMaxGrad, opt3.imSize, opt3.imSize, opt3.outX, opt3.paddingStart, 1, opt3.numChannels, 1);
			}

			else if (strcmp(opt3.layerType, "conv") == 0) {
				convWeightActs(act2PoolMax, act3Grad, weight3GradTmp, opt3.imSize, opt3.outX, opt3.outX, opt3.patchSize, opt3.paddingStart, 1, opt3.numChannels, 1, opt3.partialSum);
				weight3GradTmp.reshape(opt3.outX * opt3.outX / opt3.partialSum, opt3.numChannels * opt3.patchSize * opt3.patchSize * opt3.numFilters);
				weight3Grad.addSum(weight3GradTmp, 0, 0, 1);
				weight3Grad.reshape(opt3.numChannels * opt3.patchSize * opt3.patchSize, opt3.numFilters);
				act3Grad.reshape(opt3.numFilters, opt3.outX * opt3.outX * batchSize);
				bias3Grad.addSum(act3Grad, 1, 0, 1);
				act3Grad.reshape(opt3.numFilters * opt3.outX * opt3.outX, batchSize);

				convImgActs(act3Grad, weight3, act2PoolMaxGrad, opt3.imSize, opt3.imSize, opt3.outX, opt3.paddingStart, 1, opt3.numChannels, 1);
			}

			// 2->1
//			cout << "2->1\n";
			// original part
			act2PoolMaxGrad.transpose(false);
			act2PoolMax.transpose(false);
			act2PoolMaxGrad.eltwiseMult(dropMask2);
			act2PoolMaxGrad.scale(1.0 / keepProb);

			convCompeteOutUndo(act2PoolMaxGrad, act2PoolGrad, maxMask2, opt2.numFilters, opt2.maxOutPoolStride, opt2.poolOutX, batchSize);
			convLocalMaxUndo(act2, act2PoolGrad, act2Pool, act2Grad, opt2.poolSize, opt2.poolStartX, opt2.poolStride, opt2.poolOutX);
			convWeightActs(act1PoolMax, act2Grad, weight2GradTmp, opt2.imSize, opt2.outX, opt2.outX, opt2.patchSize, opt2.paddingStart, 1, opt2.numChannels, 1, opt2.partialSum);
			weight2GradTmp.reshape(opt2.outX * opt2.outX / opt2.partialSum, opt2.numChannels * opt2.patchSize * opt2.patchSize * opt2.numFilters);
			weight2Grad.addSum(weight2GradTmp, 0, 0, 1);
			weight2Grad.reshape(opt2.numChannels * opt2.patchSize * opt2.patchSize, opt2.numFilters);
			act2Grad.reshape(opt2.numFilters, opt2.outX * opt2.outX * batchSize);
			bias2Grad.addSum(act2Grad, 1, 0, 1);
			act2Grad.reshape(opt2.numFilters * opt2.outX * opt2.outX, batchSize);

			convImgActs(act2Grad, weight2, act1PoolMaxGrad, opt2.imSize, opt2.imSize, opt2.outX, opt2.paddingStart, 1, opt2.numChannels, 1);

			// 1->0
//			cout << "1->0\n";
			// original part
			act1PoolMaxGrad.transpose(false);
			act1PoolMax.transpose(false);
			act1PoolMaxGrad.eltwiseMult(dropMask1);
			act1PoolMaxGrad.scale(1.0 / keepProb);

			convCompeteOutUndo(act1PoolMaxGrad, act1PoolGrad, maxMask1, opt1.numFilters, opt1.maxOutPoolStride, opt1.poolOutX, batchSize);
			convLocalMaxUndo(act1, act1PoolGrad, act1Pool, act1Grad, opt1.poolSize, opt1.poolStartX, opt1.poolStride, opt1.poolOutX);
			convWeightActs(act0, act1Grad, weight1GradTmp, opt1.imSize, opt1.outX, opt1.outX, opt1.patchSize, opt1.paddingStart, 1, opt1.numChannels, 1, opt1.partialSum);
			weight1GradTmp.reshape(opt1.outX * opt1.outX / opt1.partialSum, opt1.numChannels * opt1.patchSize * opt1.patchSize * opt1.numFilters);
			weight1Grad.addSum(weight1GradTmp, 0, 0, 1);
			weight1Grad.reshape(opt1.numChannels * opt1.patchSize * opt1.patchSize, opt1.numFilters);
			act1Grad.reshape(opt1.numFilters, opt1.outX * opt1.outX * batchSize);
			bias1Grad.addSum(act1Grad, 1, 0, 1);
			act1Grad.reshape(opt1.numFilters * opt1.outX * opt1.outX, batchSize);

			// update
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


			// normalize weights

			NVNormalizeCol2(weight1, bias1, normCol1, tmp1, opt1.maxNorm);
			NVNormalizeCol2(weight2, bias2, normCol2, tmp2, opt2.maxNorm);
			NVNormalizeCol2(weight3, bias3, normCol3, tmp3, opt3.maxNorm);
			NVNormalizeCol2(weight4, bias4, normCol4, tmp4, opt4.maxNorm);
			NVNormalizeCol2(weightTop, biasTop, normColTop, tmpTop, optTop.maxNorm);


		} // for (int epoch = 0; epoch < opt1.numEpochs; epoch++)

		// compute cost
		cost /= opt1.numTrain;
		cost1 /= opt1.numTrain;
		printf("\nfinished epoch %d of %d; classify precision = %f; objective = %f; elapsed time = %f seconds\n", epoch, opt1.numEpochs,
				cost, cost1, (float)(clock() - startClock)/CLOCKS_PER_SEC);
		fprintf(pFile, "\nfinished epoch %d of %d; classify precision = %f; objective = %f; elapsed time = %f seconds\n", epoch, opt1.numEpochs,
				cost, cost1, (float)(clock() - startClock)/CLOCKS_PER_SEC);

		printf("weight norms\nweight1 = %f\nweight2 = %f\nweight3 = %f\nweight4 = %f\nweightTop = %f\n"
				"bias1 = %f\nbias2 = %f\nbias3 = %f\nbias4 = %f\nbiasTop = %f\n",
				normCol1.max(), normCol2.max(), normCol3.max(), normCol4.max(), normColTop.max(),
				bias1.norm(), bias2.norm(), bias3.norm(), bias4.norm(), biasTop.norm());
		fprintf(pFile, "weight norms\nweight1 = %f\nweight2 = %f\nweight3 = %f\nweight4 = %f\nweightTop = %f\n"
						"bias1 = %f\nbias2 = %f\nbias3 = %f\nbias4 = %f\nbiasTop = %f\n",
						normCol1.max(), normCol2.max(), normCol3.max(), normCol4.max(), normColTop.max(),
						bias1.norm(), bias2.norm(), bias3.norm(), bias4.norm(), biasTop.norm());
		printf("lr_scale = %f, mom_scale = %f, keepProb = %f, keepInputProb = %f\n", lr_scale, mom_scale, keepProb, keepInputProb);
		fprintf(pFile, "lr_scale = %f, mom_scale = %f, keepProb = %f, keepInputProb = %f\n", lr_scale, mom_scale, keepProb, keepInputProb);

		// decay learning rate
		lr_scale = lrDecay(lr_scale, opt1.lrDecayType, opt1.lrDecayFactor, opt1.lrMinRate);
		mom_scale = momInc(mom_scale, opt1.momIncType, opt1.momIncFactor, opt1.momMaxRate);
		// decay dropout
		keepProb = keepProb + opt1.keepIncRate;
		keepProb = keepProb > opt1.keepEndRate ? opt1.keepEndRate : keepProb;
		keepInputProb = keepInputProb + opt1.keepInputIncRate;
		keepInputProb = keepInputProb > opt1.keepInputEndRate ? opt1.keepInputEndRate : keepInputProb;

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
				actTop.transpose(false);
				convMaxOut(actTop, actTopMax, maxMaskTop, optTop.numFilters, optTop.maxOutPoolSize, optTop.maxOutPoolStride, 1, batchSize);
				actTopMax.transpose(true);

				//softmax layer
				NVMatrix& max = actTopMax.max(1);
				actTopMax.addVector(max, -1);
				actTopMax.apply(NVMatrixOps::Exp());
				NVMatrix& sum = actTopMax.sum(1);
				actTopMax.eltwiseDivideByVector(sum);
				delete &max;
				delete &sum;

				// compute cost
				computeLogregSoftmaxGrad(*GPURawLabelTest[batch], actTopMax, actTopMaxGrad, false, 1); // col-major
				actTopMax.transpose(false);
				computeLogregCost(*GPURawLabelTest[batch], actTopMax, trueLabelLogProbs, correctProbs); //labelLogProbs:(1, numCases); correctProbs:(1, numCases); row-major
				cost += correctProbs.sum();
				cost1 += trueLabelLogProbs.sum();

			} //for (int batch = opt1.batchNum; batch < opt1.batchNum+opt1.testBatchNum; batch++)
			cudaThreadSynchronize();
			cost /= opt1.numTest;
			cost1 /= opt1.numTest;
			printf("\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
					(float)(clock() - startClock)/CLOCKS_PER_SEC);
			fprintf(pFile, "\ntest set precision: %f\n; objective = %f; time elapsed = %f seconds\n", cost, cost1,
					(float)(clock() - startClock)/CLOCKS_PER_SEC);

			// save checkpoint
			NVSaveToFile(weight1, opt1.weightPath + "/weight1.bin");
			NVSaveToFile(weight2, opt1.weightPath + "/weight2.bin");
			NVSaveToFile(weight3, opt1.weightPath + "/weight3.bin");
			NVSaveToFile(weight4, opt1.weightPath + "/weight4.bin");
			NVSaveToFile(weightTop, opt1.weightPath + "/weightTop.bin");

			NVSaveToFile(bias1, opt1.weightPath + "/bias1.bin");
			NVSaveToFile(bias2, opt1.weightPath + "/bias2.bin");
			NVSaveToFile(bias3, opt1.weightPath + "/bias3.bin");
			NVSaveToFile(bias4, opt1.weightPath + "/bias4.bin");
			NVSaveToFile(biasTop, opt1.weightPath + "/biasTop.bin");

			printf("Checkpoint saved!\n\n");
			fprintf(pFile, "Checkpoint saved!\n\n");

		} //if (epoch % 10 == 0)

	} // for (int epoch = 0; epoch < opt1.numEpochs; epoch++)
	printf("finetune_rnorm_competeOut_maxtop() complete!\n");
	fprintf(pFile, "finetune_rnorm_competeOut_maxtop() complete!\n");

	CPUTrain.clear();
	GPUTrain.clear();
	CPUTest.clear();
	GPUTest.clear();
	GPURawLabelTrain.clear();
	GPURawLabelTest.clear();
} // int finetune_rnorm()


