
void cou_soft_max(float* input,float* output,int N){
    int max = *(std::max_element(input,input + N));
    float div = 0.f;
    for(int i = 0; i < N; i++){
        output[i] = std::exp(input[i]-M);
        div += output[i];
    }
    for(int i = 0; i < N; i++){
        output[i] /= div;
    }
}

//规约求最大值
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
//规约求和
template<unsingned int WARPSIZE>
__global__ void max_kernel(float* input,float* max_val,int N){
    __shared__ float smem[32];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = threadIdx.x / WARPSIZE;
    int laneid = threadIdx.x % WARPSIZE;

    float val = (idx < N) : input[idx] : (-FLT_MAX);

    for(int offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
        val = fmax(val,__shfl_down_sync(0xFFFFFFFF,val,offset));
    }

    if(laneid == 0){
        smem[warpid] = val;
    }
    __syncthreads();

    if(warpid == 0){
        int warpNum = blockDim.x / WARPSIZE;
        val = (laneid < WARPSIZE) ? smem[laneid] : (-FLT_MAX);
        for(int offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
            val = fmax(val,__shfl_down_sync(0xFFFFFFFF,val,offset));
        }
        if(laneid == 0){
            atomicMax(max_val,val);
        }
    }
}

__global__ void sum_kernel(float* input,float* sum,float* max_val,int N){
    __shared__ float smem[32];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = threadIdx.x / WARPSIZE;
    int laneid = threadIdx.x % WARPSIZE;

    float val = (idx < N) : expf(input[idx] - *max_val) : 0.f;

    for(int offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xFFFFFFFF,val,offset);
    }

    if(laneid == 0){
        smem[warpid] = val;
    }
    __syncthreads();

    if(warpid == 0){
        int warpNum = blockDim.x / WARPSIZE;
        val = (laneid < WARPSIZE) ? smem[laneid] : 0.f;
        for(int offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xFFFFFFFF,val,offset);
        }
        if(laneid == 0){
            atomicMax(max_val,val);
        }
    }
}
//softmax
__global__ void soft_max_kernel(float* input,float* output,float* sum,float* max_val,int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        output[idx] = expf(input[idx]-*max_val)/(*sum);
    }
}

void main(){
    int block_size = 256;
    int grid_size  = CEIL(N, block_size);
    max_kernel<<<gird_size, block_size>>>(input, max_val, N);
    sum_kernel<<<gird_size, block_size>>>(input, sum, max_val, N);
    softmax_kernel<<<gird_size, block_size>>>(input, output, sum, max_val, N);
}
