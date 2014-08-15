#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>

#include <matrix.cuh>

using namespace std;


// data is stored one record after another
void normalizeData(Matrix& data_m, char* labels, char* saveData, char* saveLabels) {
	int numRec = data_m.getNumCols();
	int numDim = data_m.getNumRows(); 

	Matrix m1;
	Matrix meanv;
 	data_m.sum(0, meanv);
	data_m.addVector(meanv, -1.0/numDim, m1);

	MTYPE pstd = 3 * sqrt(m1.norm2()/numRec/numDim);

	//m1.minWithScalar(pstd);
	//m1.maxWithScalar(-pstd);
	m1.scale(1.0/pstd);
	
	ofstream outData;
	outData.open(saveData, std::ofstream::out);
	ofstream outLabels;
	outLabels.open(saveLabels, std::ofstream::out);

	for (int i = 0; i < numRec; i++) {
		outLabels.put(labels[i]);
		for (int j = 0; j < numDim; j++) {
			outData.write((char*)&m1(j,i), sizeof(MTYPE)); // in file: data is stored record after record!
		}
	}
	outData.close();
	outLabels.close();
}
			

int main() {
	ifstream in;
	printf("starting malloc\n");
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
		in.open(full_name);
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
			data[record_start*3072+j] = MTYPE(in.get());
		}
	}
	in.close();

	printf("starting normalizing\n");
	Matrix data_m(data, 3072, 60000, true);
	normalizeData(data_m, labels, "/scratch0/qwang37/cifar-10-batches-bin/cifar_normalized.bin",
			"/scratch0/qwang37/cifar-10-batches-bin/cifar_labels.bin");
	printf("normalization successful!\n");

}








