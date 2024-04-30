
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"

int main(int argc, char* argv[])
{
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* hA;
	float* dA;
	float* hB;
	float* dB;
	float* hC;
	float* dC;

	int size = 512 * 50000;
	int  N_thread = 512;
	int N_blocks;
	int i;
	unsigned int mem_size = sizeof(float) * size;

	//hA = (float*)malloc(mem_size);
	//hB = (float*)malloc(mem_size);
	//hC = (float*)malloc(mem_size);

	cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);
	//pinned-memory

	cudaError_t err;

	err = cudaMalloc((void**)&dA, mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc((void**)&dB, mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc((void**)&dC, mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	for (i = 0; i < size; i++)
	{
		hA[i] = 1.0f / ((i + 1.0F) * (i + 1.0f));
		hB[i] = expf(1.0f / (1 + 1.0f));
		hC[i] = 0.0f;
	}

	if ((size % N_thread) == 0) {
		N_blocks = size / N_thread;
	}
	else {
		N_blocks = (int)(size / N_thread) + 1;
	}
	dim3 blocks(N_blocks);

	cudaEventRecord(start, 0);

	err = cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot copy data host/device : %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot copy data host/device : %s\n", cudaGetErrorString(err));
		return 1;
	}



	function << < N_blocks, N_thread >> > (dA, dB, dC, size);

	

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot copy data device/host : %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);


	cudaEventRecord(start, 0);

	for (i = 0; i < size; i++)
	{
		hC[i] = sinf(sinf(hA[i] * hB[i]));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);

	printf("\n Rate: %f x\n", timerValueCPU / timerValueGPU);

	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;
}

