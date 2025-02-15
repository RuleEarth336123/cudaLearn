#define CEIL(m,n) (m + n -1)/n

__global__ void reduce_mean_kernel(double* input,double* output,int N){
    
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double smem[32] = {0.f};
    const int warpSize = 32;
    const int warpid = threadIdx.x / warpSize;
    const int laneid = threadIdx.x % warpSize;

    double val = (idx < N) : input[i] / N : 0.f;

    for(int offset >> warpSize/2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xFFFFFFFF,val,offset);
    }

    if(laneid == 0){
        smem[warpid] = val;
    }
    __syncthreads();

    if(warpid == 0){
        const int warpNum = blockDimx.x / warpSize;
        val = (warpid < warpNum) ? smem[laneid] : 0.f;
        for(int offset >> warpSize/2; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xFFFFFFFF,val,offset);
        }
        if(laneid == 0){
            atomicAdd(output,val); 
        }
    }
}

__global__ void reduce_variance_kernel(double* input,double* mean,double* output,int N){
    
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double smem[32] = {0.f};
    const int warpSize = 32;
    const int warpid = threadIdx.x / warpSize;
    const int laneid = threadIdx.x % warpSize;

    double val = (idx < N) : pow((input[i] - *mean),2)/N : 0.f;

    for(int offset >> warpSize/2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xFFFFFFFF,val,offset);
    }

    if(laneid == 0){
        smem[warpid] = val;
    }
    __syncthreads();
    if(warpid == 0){
        const int warpNum = blockDimx.x / warpSize;
        val = (warpid < warpNum) ? smem[laneid] : 0.f;
        for(int offset >> warpSize/2; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xFFFFFFFF,val,offset);
        }
        if(laneid == 0){
            atomicAdd(output,val); 
        }
    }
}

template<double gamma,double beta>
__global__ void layernorm(double* input,double* output,double* mean,double* variance,double* u,int N){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        output[i] = gamma * (input[i] - mean)/sqrt(*variance) + beta; 
    }
}

void main1(){

    const int N = 10240000;

    double* input_h = (double*)malloc(N * (sizeof(double)));
    double* output_h = (double*)malloc(N * (sizeof(double)));
    double* mean_h = (double*)malloc(N * (sizeof(double)));
    double* variance_h = (double*)malloc(N * (sizeof(double)));

    double* input_d;
    double* output_d;
    double* mean_d;
    double* variance_d;

    cudaMalloc((void**)&input_d,N * (sizeof(double)));
    cudaMalloc((void**)&output_d,N * (sizeof(double)));
    cudaMalloc((void**)&variance_d,N * (sizeof(double)));
    cudaMalloc((void**)&mean_d,N * (sizeof(double)));
    
    const int BLOCKSIZE = 256;
    const int GRIDSIZE = CEIL(N,BLOCKSIZE);

    dim3 gird(GRIDSIZE);
    dim3 block(BLOCKSIZE); 

    cudaMemcpy(input_d,input_h,cudaMemcpyDeviceToHost);
    cudaMemset(mean_d,0,sizeof(double));
    cudaMemset(variance_d,0,sizeof(double));
    cudaMemset(output_d,0,sizeof(double));


    reduce_mean_kernel<<<grid,block>>>(input_d,variance_d,N);
    reduce_variance_kernel<<<grid,block>>>(input,mean,variance_d,N);
    layernorm<1.0,0.0><<<grid,block>>>(input_d,output_d,mean_d,variance_d,N);

    cudaMemcpy(output_h, output_d, N * sizeof(double), cudaMemoryDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(mean_d);
    cudaFree(variance_d);

    free(input_h);
    free(output_h);
    free(mean_h);
    free(variance_h);
}

//算子融合
template <double gamma,double beta>
__global__ void fused_layernorm_kernel(double* input,double* output,int N){
    extern __shared__ double smem[];
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int warpSize = 32;
    const int warpNum = blockDim.x / warpSize;
    const int warpid = threadIdx.x / warpSize;
    const int laneid = threadIdx.x % warpSize;

    double val_mean = (idx < N) : input[i] / N : 0.f;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val_mean += __shfl_down_sync(0xFFFFFFFF, val_mean, offset);
    }
    if (laneid == 0) {
        smem[warpid] = val_mean;
    }
    __syncthreads();

    if(warpid == 0){
        const int warpNum = blockDim.x / warpSize;
        val_mean = (laneid < warpSize) ? smem[laneid] : 0.0;
        for(int offset = warpSize / 2; offset > 0; offset >>= 1){
            val_mean += __shfl_down_sync(0xFFFFFFFF,val_mean,offset);
        }
        if(laneid == 0){
            atomicAdd(&smem[warpNum],val_mean);
        }
    }
    __syncthreads();


    double mean = smem[blockDim.x / warpSize];

    double var_variance = (idx < N) > pow(input[idx] - mean,2) / N : 0.0;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        var_variance += __shfl_down_sync(0xFFFFFFFF, var_variance, offset);
    }
    if (laneid == 0) {
        smem[warpid] = var_variance;
    }
    __syncthreads();

    if(warpid == 0){
       
        val_mean = (laneid < warpSize) ? smem[laneid] : 0.0;
        for(int offset = warpSize / 2; offset > 0; offset >>= 1){
            val_mean += __shfl_down_sync(0xFFFFFFFF,val_mean,offset);
        }
        if(laneid == 0){
            atomicAdd(&smem[warpNum + 1],val_mean);
        }
    }
    __syncthreads();

    double variance = smem[warpNum + 1]; 

    if (idx < N) {
        output[idx] = gamma * (input[idx] - mean) / sqrt(variance) + beta;
    }

}

int main2() {
    const int N = 10240000;

    double* input_h = (double*)malloc(N * sizeof(double));
    double* output_h = (double*)malloc(N * sizeof(double));

    double* input_d;
    double* output_d;

    cudaMalloc((void**)&input_d, N * sizeof(double));
    cudaMalloc((void**)&output_d, N * sizeof(double));

    // Initialize input_h with some data
    for (int i = 0; i < N; i++) {
        input_h[i] = static_cast<double>(i);
    }

    cudaMemcpy(input_d, input_h, N * sizeof(double), cudaMemcpyHostToDevice);

    const int BLOCKSIZE = 256;
    const int GRIDSIZE = CEIL(N, BLOCKSIZE);

    dim3 grid(GRIDSIZE);
    dim3 block(BLOCKSIZE);

    // Calculate shared memory size
    size_t shared_mem_size = (BLOCKSIZE / 32 + 2) * sizeof(double); // For mean and variance

    // Launch fused kernel
    const double gamma = 1.0;
    const double beta = 0.0;
    fused_layernorm_kernel<gamma, beta><<<grid, block, shared_mem_size>>>(input_d, output_d, N);

    cudaMemcpy(output_h, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Print some results for verification
    std::cout << "First 10 output values:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output_h[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(input_d);
    cudaFree(output_d);

    // Free host memory
    free(input_h);
    free(output_h);

    return 0;
}