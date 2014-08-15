// test blocks
#ifndef TEST_CUH_
#define TEST_CUH_

void testSamplePatches();
void testMult();
void testConv();
void testMatrixIO();
void testDataIO();
void testTrainFCAE();
void testGenerateDataConv(char* poolType);
void testGenerateDataFC();
void testNVLabelReadFromFile();
void testNVRawLabelReadFromFile();

/*
void trainLayer1();
void trainLayer2();
void trainLayer3();
void generateLayer2Data();
void generateLayer3Data();
*/

//void finetune();
//void finetune1();
void finetune_rnorm();
void finetune_rnorm_side();
void finetune_rnorm_side_append();
//void finetune_rnorm_side_5layers();
void finetune_alter_update();
void finetune_rnorm_dropConnect();
void finetune_rnorm_dropOut();
void finetune_rnorm_dropPerturb();
void finetune_rnorm_dropUpdate(int startDrop, int interval);
void finetune_rnorm_dropPerturb_3conv();
void finetune_rnorm_threshold();
void finetune_rnorm_dualRelu();
void finetune_rnorm_maxout();
void finetune_rnorm_competeOut();
void finetune_rnorm_competeTop();
void finetune_rnorm_maxout_perturb();
void finetune_rnorm_maxout_dropout_perturb();
void finetune_rnorm_competeOut_dropout_perturb();
void finetune_rnorm_competeAbs();
void finetune_rnorm_competeOut_maxtop();

void multiViewTest();
void multiViewTest_side();
void multiViewTest_side_append();
//void multiViewTest_side_5layers();
void multiViewTest_dropOut();
void multiViewTest_dropPerturb();
void multiViewTest_dropUpdate();
void multiViewTest_dropPerturb_3conv();
void multiViewTest_maxout();
//void multiViewTest_competeOut();
void extractAct(string actPath);
void extractActMaxout(string actPath);

void testCropDataProvider();
void testNVReadFromFileUint8();

void centerData();
void convertToMTYPE();
void testAssembleMatrix();
void testAssembleMatrix1();
void testAssembleMatrix2();
void testGenFilterMask();
void testAbs();



#endif
