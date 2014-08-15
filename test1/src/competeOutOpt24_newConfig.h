//#ifndef RNORMOPT_H
//#define RNORMOPT_H

#include "opt.cuh"
#include <stdio.h>

#ifndef OPT1
#define OPT1
LayerOpt opt1, opt2, opt3, opt4, opt5, optTop;
MTYPE scale1, scale2, scale3, scale4;

#endif

void assignOpt() {
	strcpy(opt1.layerName, "layer1");
	opt1.labelSize = 10;
	opt1.imSize = 24;
	opt1.patchSize = 8;
	opt1.numChannels = 3;
	opt1.numVis = opt1.patchSize * opt1.patchSize * opt1.numChannels;
	opt1.numFilters = 80;
	opt1.paddingStart = -4;
	opt1.batchSize = 128;
	opt1.numTrain = 50000;
	opt1.numTest = 10000;
	opt1.numEpochs = 350;
	opt1.initstv = 0.005;
	opt1.mom = 0.5;
	opt1.lrW = 0.05;
	opt1.lrB = 0.005;
	opt1.weightDecay = 0.00;
	opt1.sparseParam = 0.035;
	opt1.sparseWeight = 0.000;
	// pool
	opt1.poolSize = 4;
	opt1.poolStride = 2;
	opt1.poolStartX = 0;
	opt1.outX = opt1.imSize - 2 * opt1.paddingStart - opt1.patchSize + 1;
	opt1.poolOutX = (opt1.outX - opt1.poolSize - opt1.poolStartX) / opt1.poolStride + 2;
	strcpy(opt1.pooler, "max");

	strcpy(opt1.neuronType, "relu");
	strcpy(opt1.layerType, "conv");
	// side parameters
	opt1.numFilters_side = 32;
	// rnorm parameters
	opt1.sizeF = 5; // size of the rnorm neighborhood
	opt1.addScale = 0.000; // alpha value in the normalization equation
	opt1.powScale = 0.75; // power parameter in the normalization equation
	// affect speed
	opt1.partialSum = 1;
	opt1.numViews = 1; // number of views for multi-view test
	opt1.loadParam = false;
	opt1.maxOutPoolSize = 4;
	opt1.maxOutPoolStride = 4;
	opt1.numGroups = (opt1.numFilters - 1) / opt1.maxOutPoolStride + 1;
	opt1.dataPath.assign("/scratch0/qwang37/cifar-10-batches-bin");
	opt1.weightPath.assign("/scratch0/qwang37/weights_10_7_augment");
	opt1.exp.assign("cifar10");
	// weight regularization
	opt1.maxNorm = 0.9;
	strcpy(opt1.lrDecayType, "exponential");
	strcpy(opt1.momDecayType, "linear");
	opt1.lrStartScale = 1.0;
	opt1.momStartScale = 1.0;
	opt1.lrDecayFactor = 0.95;
	opt1.momDecayFactor = 0.0;
	opt1.lrMinRate = 0.01;
	opt1.momMinRate = 0.1;

	opt1.whitened = true;
	opt1.inputScale = 2.539;
	opt1.keepProbInput = 1.0;

	strcpy(opt2.layerName, "layer2");
	opt2.imSize = opt1.poolOutX;
	opt2.patchSize = 8;
	opt2.numChannels = opt1.numFilters;
	opt2.numVis = opt2.patchSize * opt2.patchSize * opt2.numChannels;
	opt2.numFilters = 176;
	opt2.paddingStart = -3;
	opt2.initstv = 0.005;
	opt2.mom = opt1.mom;
	opt2.lrW = 0.05;
	opt2.lrB = 0.005;
	opt2.weightDecay = 0.00;
	opt2.sparseParam = 0.035;
	opt2.sparseWeight = 0.0;
	//pool
	opt2.poolSize = 4;
	opt2.poolStride = 2;
	opt2.poolStartX = 0;
	opt2.outX = opt2.imSize - 2 * opt2.paddingStart - opt2.patchSize + 1;
	opt2.poolOutX = (opt2.outX - opt2.poolSize - opt2.poolStartX) / opt2.poolStride + 2;
	strcpy(opt2.pooler, "max");

	strcpy(opt2.neuronType, "relu");
	strcpy(opt2.layerType, "conv");
	opt2.partialSum = 1;
	// side parameters
	opt2.numFilters_side = 32;
	// rnorm parameters
	opt2.sizeF = 5; // size of the rnorm neighborhood
	opt2.addScale = 0.000; // alpha value in the normalization equation
	opt2.powScale = 0.75; // power parameter in the normalization equation
	//maxout
	opt2.maxOutPoolSize = 4;
	opt2.maxOutPoolStride = 4;
	opt2.numGroups = (opt2.numFilters - 1) / opt2.maxOutPoolStride + 1;
	opt2.maxNorm = 3.9;

	opt2.inputScale = 1.0f;

	strcpy(opt3.layerName, "layer3");
	opt3.imSize = opt2.poolOutX;
	opt3.patchSize = 5;
	opt3.numChannels = opt2.numFilters;
	opt3.numVis = opt3.patchSize * opt3.patchSize * opt3.numChannels;
	opt3.numFilters = 176;
	opt3.paddingStart = -3;
	opt3.initstv = 0.005;
	opt3.mom = opt1.mom;
	opt3.lrW = 0.05;
	opt3.lrB = 0.005;
	opt3.weightDecay = 0.00;
	opt3.sparseParam = 0.035;
	opt3.sparseWeight = 0.0;
	// pool
	opt3.poolSize = 2;
	opt3.poolStride = 2;
	opt3.poolStartX = 0;
	opt3.outX = opt3.imSize - 2 * opt3.paddingStart - opt3.patchSize + 1;
	opt3.poolOutX = (opt3.outX - opt3.poolSize - opt3.poolStartX) / opt3.poolStride + 2;

	strcpy(opt3.neuronType, "relu");
	strcpy(opt3.layerType, "conv");
	opt3.partialSum = 1;
	// side parameters
	opt3.numFilters_side = 32;
	//maxout
	opt3.maxOutPoolSize = 4;
	opt3.maxOutPoolStride = 4;
	opt3.numGroups = (opt3.numFilters - 1) / opt3.maxOutPoolStride + 1;
	opt3.maxNorm = 3.9;

	opt3.inputScale = 1.0f;

	strcpy(opt4.layerName, "layer4");
	opt4.imSize = opt3.poolOutX;
	opt4.patchSize = 3;
	opt4.numChannels = opt3.numFilters;
	opt4.numVis = opt3.poolOutX * opt3.poolOutX * opt3.numFilters;
	opt4.numFilters = 1210;
	opt4.paddingStart = -1;
	opt4.initstv = 0.005;
	opt4.mom = opt1.mom;
	opt4.lrW = 0.1;
	opt4.lrB = 0.01;
	opt4.weightDecay = 0.00;
	opt4.sparseParam = 0.035;
	opt4.sparseWeight = 0.0;
	//opt4.outX = opt4.imSize - 2 * opt4.paddingStart - opt4.patchSize + 1;
	opt4.outX = 11;

	strcpy(opt4.neuronType, "relu");
	strcpy(opt4.layerType, "local");
	// side parameters
	opt4.numFilters_side = 32;
	opt4.maxOutPoolSize = 10;
	opt4.maxOutPoolStride = 10;
	opt4.numGroups = (opt4.numFilters/opt4.outX/opt4.outX - 1)/ opt4.maxOutPoolStride + 1;
	//opt4.maxNorm = 1.9 * 208.0 / 128.0;
	opt4.maxNorm = 3.9;

	opt4.inputScale = 1.0f;

	optTop.numFilters = 10; // classification nodes
	optTop.initstv = 0.005;
	optTop.mom = opt1.mom;
	optTop.lrW = 0.1;
	optTop.lrB = 0.01;
	optTop.weightDecay = 0.00;
	optTop.sparseParam = 0.035;
	optTop.sparseWeight = 0.0;
	optTop.numVis = opt4.numFilters;
	strcpy(optTop.neuronType, "");
	strcpy(optTop.layerType, "softmax");
	optTop.maxNorm = 5.9;

	optTop.inputScale = 1.0f;
}

//#endif
