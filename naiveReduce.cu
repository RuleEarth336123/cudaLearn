#include <stdio.h>
#include "time.h"
#include <cuda.h>
#include "device_launch_parameters.h"

int reduce(int *arr, int len){
	int sum = 0;
    for(int i = 0;i < len; i++){
        sum += arr[i];
    }
}

template<int BLOCKSIZE>
__global__ void reduce_naive_kernel(int *arr,int* out,int len){
    
    __shared__ int sdata[BLOCKSIZE];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //将数据拷贝到共享内存
    if(idx < len){
        sdata[idx] = arr[idx];
    }
    __syncthreads();

    for(int i = 1; i < blockDim.x; i *= 2){
        if(tid % (2 * i) == 0 && idx + i < len){
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    //将block的第一个线程写入out中
    if(tid == 0){
        out[blockIdx.x] = sdata[0];
    }
}


int main() {
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	printf("device : %s start\n", deviceProp.name);

	cudaSetDevice(dev);

	bool bResult = false;

	int size = 1 << 24;	//归约的元素数
	int blockSize = 32;
    int gridSize = (size + blockSize - 1) / blockSize;

	dim3 block(blockSize,1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	printf("block:%d  gird:%d\n", block.x, grid.x);

	int* h_idata = (int*)malloc(sizeof(int) * size);
	int* h_odata = (int*)malloc(sizeof(int) * grid.x);

	for (int i = 0; i < size; i++) {
		h_idata[i] = (int)(rand() & 0xFF);
	}

	int cpu_sum = 0;
	for (int i = 0; i < size; i++) {
		cpu_sum += h_idata[i];
	}

	int gpu_sum = 0;
	int* d_idata = NULL;
	int* d_odata = NULL;
	cudaMalloc((void**)&d_idata, sizeof(int) * size);
	cudaMalloc((void**)&d_odata, sizeof(int) * grid.x);

	//执行kernel1
	cudaMemcpy(d_idata, h_idata, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
    reduce_naive_kernel<32> << <gridSize, blockSize >> > (d_idata, d_odata, size);
	cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	for (int i = 0; i < gridSize; i++) {
		gpu_sum += h_odata[i];
	}
	cudaDeviceSynchronize();

	free(h_idata);
	free(h_odata);
	cudaFree(d_idata);
	cudaFree(d_odata);

	cudaDeviceReset();

    printf("cpu sum = %d\n",cpu_sum);
    printf("gpu sum = %d\n",gpu_sum);

	if (gpu_sum == cpu_sum) {
		printf("Test PASSED\n");
	}
	else {
		printf("Test FAILED\n");
	}

	return 0;

}
