//每个block中有一个warp，每个warp负责一行的计算。
__global__ void sgemv(float* matrix,float* vector,float* out_vec,int M,int K){

    int laneid = threadIdx.x % WARPSIZE;
    int row = blockIdx.x;
    if(row > M){
        return;
    }

    float res = 0.f;
    int kit = CEIL(K,WARPSIZE);// 每个线程需要负责计算的数据个数

    for(int i = 0; i < kit; i++){
        int col = i * WARPSIZE + laneid;
        res += (col < K) ? A[row * K + col] * x[col] : 0.f;
    }

    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        res += __shfl_down_sync(0xFFFFFFFF,res,offset);
    }

    if(lanid == 0){
        y[row] = res;
    }
}