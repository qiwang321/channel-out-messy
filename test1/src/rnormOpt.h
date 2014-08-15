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
	opt1.patchSize = 5;
	opt1.numChannels = 3;
	opt1.numFilters = 64;
	opt1.paddingStart = -2;
	opt1.batchSize = 128;
	opt1.numTrain = 40000;
	opt1.numTest = 10000;
	opt1.numEpochs = 350;
	opt1.initstv = 0.0001;
	opt1.mom = 0.9;
	opt1.lrW = 0.001;
	opt1.lrB = 0.002;
	opt1.weightDecay = 0.000;
	opt1.sparseParam = 0.035;
	opt1.sparseWeight = 0.000;
	opt1.poolSize = 3;
	opt1.poolStride = 2;
	opt1.poolStartX = 0;
	opt1.outX = opt1.imSize - 2 * opt1.paddingStart - opt1.patchSize + 1;
	opt1.poolOutX = (opt1.outX - opt1.poolSize - opt1.poolStartX) / opt1.poolStride + 2;
	strcpy(opt1.pooler, "max");
	opt1.numVis = opt1.patchSize * opt1.patchSize * opt1.numChannels;
	strcpy(opt1.neuronType, "relu");
	strcpy(opt1.layerType, "conv");
	// side parameters
	opt1.numFilters_side = 32;
	// rnorm parameters
	opt1.sizeF = 9; // size of the rnorm neighborhood
	opt1.addScale = 0.001; // alpha value in the normalization equation
	opt1.powScale = 0.75; // power parameter in the normalization equation
	// affect speed
	opt1.partialSum = 4;
	opt1.numViews = 10; // number of views for multi-view test
	opt1.loadParam = false;

	strcpy(opt2.layerName, "layer2");
	opt2.imSize = 12;
	opt2.patchSize = 5;
	opt2.numChannels = 64;
	opt2.numFilters = 64;
	opt2.paddingStart = -2;
	opt2.initstv = 0.01;
	opt2.mom = opt1.mom;
	opt2.lrW = opt1.lrW;
	opt2.lrB = opt1.lrB;
	opt2.weightDecay = 0.000;
	opt2.sparseParam = 0.035;
	opt2.sparseWeight = 0.0;
	opt2.poolSize = 3;
	opt2.poolStride = 2;
	opt2.poolStartX = 0;
	opt2.outX = opt2.imSize - 2 * opt2.paddingStart - opt2.patchSize + 1;
	opt2.poolOutX = (opt2.outX - opt2.poolSize - opt2.poolStartX) / opt2.poolStride + 2;
	strcpy(opt2.pooler, "max");
	opt2.numVis = opt2.patchSize * opt2.patchSize * opt2.numChannels;
	strcpy(opt2.neuronType, "relu");
	strcpy(opt2.layerType, "conv");
	// side parameters
	opt2.numFilters_side = 32;
	// rnorm parameters
	opt2.sizeF = 9; // size of the rnorm neighborhood
	opt2.addScale = 0.001; // alpha value in the normalization equation
	opt2.powScale = 0.75; // power parameter in the normalization equation
	opt2.partialSum = 8;

	strcpy(opt3.layerName, "layer3");
	opt3.imSize = 6;
	opt3.patchSize = 3;
	opt3.numChannels = 64;
	opt3.numFilters = 64;
	opt3.paddingStart = -1;
	opt3.initstv = 0.04;
	opt3.mom = opt1.mom;
	opt3.lrW = opt1.lrW;
	opt3.lrB = opt1.lrB;
	opt3.weightDecay = 0.004;
	opt3.sparseParam = 0.035;
	opt3.sparseWeight = 0.0;
	opt3.outX = opt3.imSize - 2 * opt3.paddingStart - opt3.patchSize + 1;
	opt3.numVis = opt3.patchSize * opt3.patchSize * opt3.numChannels;
	strcpy(opt3.neuronType, "relu");
	strcpy(opt3.layerType, "local");
	// side parameters
	opt3.numFilters_side = 32;

	strcpy(opt4.layerName, "layer4");
	opt4.imSize = 6;
	opt4.patchSize = 3;
	opt4.numChannels = 64;
	opt4.numFilters = 32;
	opt4.paddingStart = -1;
	opt4.initstv = 0.04;
	opt4.mom = opt1.mom;
	opt4.lrW = opt1.lrW;
	opt4.lrB = opt1.lrB;
	opt4.weightDecay = 0.004;
	opt4.sparseParam = 0.035;
	opt4.sparseWeight = 0.0;
	opt4.outX = opt4.imSize - 2 * opt4.paddingStart - opt4.patchSize + 1;
	opt4.numVis = opt4.patchSize * opt4.patchSize * opt4.numChannels;
	strcpy(opt4.neuronType, "relu");
	strcpy(opt4.layerType, "local");
	// side parameters
	opt4.numFilters_side = 32;


	optTop.numFilters = 10; // classification nodes
	optTop.initstv = 0.01;
	optTop.mom = opt1.mom;
	optTop.lrW = opt1.lrW;
	optTop.lrB = opt1.lrB;
	optTop.weightDecay = 0.01;
	optTop.sparseParam = 0.035;
	optTop.sparseWeight = 0.0;
	optTop.numVis = opt4.outX * opt4.outX *opt4.numFilters;
	strcpy(optTop.neuronType, "");
	strcpy(optTop.layerType, "softmax");
}

//#endif
