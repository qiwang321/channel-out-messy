################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/competeOut.cu \
../src/competeOut_maxtop.cu \
../src/extractAct.cu \
../src/layer_kernels.cu \
../src/main.cu \
../src/maxout.cu \
../src/myClass.cu \
../src/routines.cu \
../src/tests.cu 

CU_DEPS += \
./src/competeOut.d \
./src/competeOut_maxtop.d \
./src/extractAct.d \
./src/layer_kernels.d \
./src/main.d \
./src/maxout.d \
./src/myClass.d \
./src/routines.d \
./src/tests.d 

OBJS += \
./src/competeOut.o \
./src/competeOut_maxtop.o \
./src/extractAct.o \
./src/layer_kernels.o \
./src/main.o \
./src/maxout.o \
./src/myClass.o \
./src/routines.o \
./src/tests.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/home/qiwang321/nvmatrix_test/include -I/home/qiwang321/nvmatrix_test/include/common -I/home/qiwang321/nvmatrix_test/include/cudaconv2 -I/home/qiwang321/nvmatrix_test/include/nvmatrix -O3 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I/home/qiwang321/nvmatrix_test/include -I/home/qiwang321/nvmatrix_test/include/common -I/home/qiwang321/nvmatrix_test/include/cudaconv2 -I/home/qiwang321/nvmatrix_test/include/nvmatrix -O3 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


