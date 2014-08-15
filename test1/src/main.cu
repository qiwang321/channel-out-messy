#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <nvmatrix.cuh>
#include "opt.cuh"
#include <cudaconv2.cuh>

// convnet routines
#include "routines.cuh"
// test blocks
#include "tests.cuh"
//#include "competeOutOpt32_cifar100.h"
//#include "competeOutOpt32_cifar100_pure.h"
//#include "competeOutOpt32_cifar100_newConfig.h"
//#include "competeOutOpt32_cifar100_original.h"
//#include "competeOutOpt32_cifar100_newConfig_vary.h"
//#include "competeOutOpt32_cifar10_newConfig_vary.h"
//#include "competeOutOpt32_stl10_newConfig_vary.h"
//#include "maxOutOpt32_cifar10_newConfig_vary.h"
//#include "maxOutOpt32_cifar100_newConfig_vary.h"

//#include "maxOutOpt32_cifar100_10.h"
//#include "competeOutOpt32_cifar100_10.h"

//#include "competeOutOpt32_cifar100_20.h"
//#include "maxOutOpt32_cifar100_20.h"

//#include "competeOutOpt32_cifar100_50.h"
//#include "maxOutOpt32_cifar100_50.h"

//#include "competeOutOpt32_cifar100_100.h"
//#include "maxOutOpt32_cifar100_100.h"
//#include "competeOutOpt32_cifar10_max_top.h"
#include "competeOutOpt24_cifar10_max_top.h"

using namespace std;

FILE* pFile;
//float keepProb = 4.0 / 8;
//float updateProb = 1.0;
float perturbScale = 0.0; // for the model perturbation exp
//float widerScale = 0.0;

int main() {

	// initialize cublas
	int deviceId = 0;
	cudaSetDevice(deviceId); //////////////////////careful when doing experiments on different platforms ///////////////////////////
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cublasInit();

	assignOpt();

	string log_path = opt1.weightPath + "/log_10_25_cifar10_maxtop.txt";
	pFile = fopen(log_path.c_str(), "a");

	printf("test 10/25/2013: %s\nkeepProb = %f\nperturbScale = %f\nexp = %s\ndevice = %d\n\n", opt1.testName, opt1.keepStartRate, perturbScale, opt1.exp.c_str(), deviceId);
	fprintf(pFile, "test 10/25/2013: %s\nkeepProb = %f\nperturbScale = %f\nexp = %s\ndevice = %d\n\n", opt1.testName, opt1.keepStartRate, perturbScale, opt1.exp.c_str(), deviceId);
	printf("log info saving to %s\n", log_path.c_str());
	fprintf(pFile, "log info saving to %s\n", log_path.c_str());

	srand(time(0));
	NVMatrix::initRandom();


	//scaleWeights5(opt1.weightPath, 0.8f);

	assignOpt();
	printf("stage 1...\n");
	fprintf(pFile, "stage 1...\n");
	opt1.loadParam = false;
	opt1.whitened = true;

	//finetune_rnorm_maxout();
	//finetune_rnorm_competeOut();
	finetune_rnorm_competeOut_maxtop();


//	extractAct(opt1.weightPath);
//	extractActMaxout(opt1.weightPath);




	printf("test 10/25/2013 completed: %s\nkeepProb = %f\nperturbScale = %f\nexp = %s\ndevice = %d\n", opt1.testName, opt1.keepStartRate, perturbScale, opt1.exp.c_str(), deviceId);
	fprintf(pFile, " test 10/25/2013 completed: %s\nkeepProb = %f\nperturbScale = %f\nexp = %s\ndevice = %d\n", opt1.testName, opt1.keepStartRate, perturbScale, opt1.exp.c_str(), deviceId);
	printf("log info saved to %s\n", log_path.c_str());
	fprintf(pFile, "log info saved to %s\n", log_path.c_str());
	fclose(pFile);

	NVMatrix::destroyRandom();

}
