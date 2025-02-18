//每个block中有一个warp，每个warp负责一行的计算。
__global__ void sgemv(float* input_m,float* input_v,float* out_v,int M,int K){

    __shared__ float smem_v[32];// 共享内存缓存向量 x

    int laneid = threadIdx.x % warpSize;
    int row = blockIdx.x;
    if(row > M){
        return;
    }

    float res = 0.f;
    int kit = CEIL(K,warpSize);// 每个线程需要负责计算的数据个数

    // 将向量 x 加载到共享内存
    for(int i = 0; i < kit; i++){
        int col = i * WARPSIZE + laneid;

        s_x[laneid] = (col < K) ? input_v[col] : 0.f;
        __syncthreads();

        for(int j = 0; j < warpSize; j++){
            col = i * warpSize + j;
            res += (col < K) ? input_m[row * K + col] * smem_v[j] : 0.f;
        }
        __syncthreads();
    }
    // Warp 内归约求和
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        res += __shfl_down_sync(0xFFFFFFFF,res,offset);
    }

    if(lanid == 0){
        out_v[row] = res;
    }
}