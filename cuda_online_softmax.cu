#include "cuda_runtime.h"

#define VECTORIZER_LOAD(vec_ptr, local_val, idx)\
    reinterpret_cast<float4*>(&local_val)[0] = reinterpret_cast<float4*>(vec_ptr)[(idx)/4]


__global__ void online_softmax_kernel(float* input,float* output, int rows){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        const float* input_start = input + idx * rows;
        float* output_start = output + idx * rows;

        float maxval = -INFINITY;
        double sum = 0.0;
        for(int i = 0; i < rows; i++){
            float maxval_prev = maxval;
            if(input_start[i] > maxval){
                maxval = input_start[i];
                sum = sum * expf(maxval_prev - max_val) +  expf(input_start[i] - max_val);
            }else{
                sum += expf(input_start[j] - max_val);
            }
        }

        for(int i = 0; i < rows; i++){
            output_start[i] = expf(input_start[i] - max_val) / sum;
        }
    }
}

struct __align__(8) MD{
    float dmax;
    float dsum;
};
// “合并”两个 MD 结构，得到同时代表这两部分数据的最大值和归一化因子。其核心思路
__device__ __forceinline__ MD reduce_md_op(MD a, MD b){
    bool a_bigger = a_bigger ? a : b;
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;

    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m-bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void online_softmax_kernel_v2(const float* __restrict input, float* __restrict output, int V){
    const int tid = threadIdx.x;
    
    float* input_start = input + blockIdx.x * V;
    float* output_start = output + blockIdx.x * V;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;// 共享内存存储临时归约数据
    __shared__ MD shared_md;            // 存储全局的max和sum_exp

    // 每个线程计算局部MD值
    MD md_partial;
    md_partial.dmax = -FLT_MAX;
    md_partial.dsum = 0.f;
    
    for(int i = tid; i < V; i += THREADBLOCK_SIZE){
        MD new_elem;
        new_elem.dmax = input_start[i];
        new_elem.dsum = 1.f;
        float val = input_start[i];
        md_partial = reduce_md_op(md_partial, reduce_md_op);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if(tid == 0){
        shared_md = md;
    }
    __syncthreads();

    float d_total_inverse = __fdividef(1.f, md_total.dsum);
    for(int i = tid; i < V; i += THREADBLOCK_SIZE){
        output_start[i] = __expf(input_start[i] - md.dmax) * d_total_inverse;
    }
}


__global__ void online_softmax_kernel_v0(const float* input, float* output, int N){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;
    const int warpid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    float max_val = -FLT_MAX;
    float sum_exp = 0.f;

    for(int i = 0; i < N; i++){
        
    }
}