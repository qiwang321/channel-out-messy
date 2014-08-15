#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "matrix.cuh"

using namespace std;

void samplePatches(int patchSize, int numPatches, int dims[], int numRecords, char in_name[], char out_name[]) {
	int patchesPerRecord = numPatches / numRecords;
	ifstream in;
	in.open(in_name, std::ifstream::in | std::ifstream::binary);
	if (in.fail()) {
		printf("data file open failed!\n");
		exit(-1);
	}
	ofstream out;
	out.open(out_name, std::ofstream::out | std::ofstream::binary);
	if (out.fail()) {
		printf("creating output file failed!\n");
		exit(-1);
	}
	int dimall = dims[0]*dims[1]*dims[2];
	int dim2 = dims[0]*dims[1];
	MTYPE* data = (MTYPE*) malloc(dimall*sizeof(MTYPE));

	for (int i = 0; i < numRecords; i++) {
		in.read((char*)data, dimall*sizeof(MTYPE));
		for (int j = 0; j < patchesPerRecord; j++) {
			// data is row-major: pixels->channels->images
			int pixelX = rand() % (dims[0] - patchSize + 1);
			int pixelY = rand() % (dims[1] - patchSize + 1);
			for (int c = 0; c < dims[2]; c++)
				for (int y = 0; y < patchSize; y++)
					for (int x = 0; x < patchSize; x++)
						out.write((char*)(data + i*dimall + c*dim2
								+ (pixelY+y)*dims[0] + pixelX + x), sizeof(MTYPE));
		}
	}
	in.close();
	out.close();
}

int main() {
	char* in_name = "/scratch0/qwang37/cifar-10-batches-bin/cifar_normalized.bin";
	char* out_name = "/scratch0/qwang37/cifar-10-batches-bin/cifar_patches.bin";
	int dims[3] = {32, 32, 3};
	samplePatches(5, 2000000, dims, 50000, in_name, out_name);
	printf("patch sampling successfully finished!");

}
