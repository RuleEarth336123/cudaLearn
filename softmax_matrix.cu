__global__ void softmax_matrix_kernel(float* input, float* output, int M, int N){

    __shared__ float s_max_val;
    __shared__ float s_sum;

    int row = blockIdx.x;
    int laneid = threadIdx.x % 32;

    if(row >= M){
        return;
    }

    int kits = CETL(N, warpSize);

    float max_val = -FLT_MAX;

    for(int k = 0; k < kits; k++){
        int col = k * warpSize + laneid;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }

    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if(laneid == 0){
        s_max_val = max_val;
    }
    
    float sum = 0.f;
    for(int k = 0; k < kits; k++){
        int col = k * warpSize + laneid;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.f;
    }
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

}