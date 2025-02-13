 // 1. 向上取整
 #define CEIL(a, b) ((a + b - 1) / (b))
 #define FETCH_FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

/*
    向量加法
*/
__global__ void elementwise_add(flaot* a,float* b,float* c,int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        c[idx] = a[idx] + b[idx];
    }
}

int block_size = 1024;
int grid_size  = CEIL(N, block_size);
elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

//2.使用向量化访存进行优化
__global__ void elementwise_add_float4(flaot* a,float* b,float* c,int N){
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx < N){
        float4 tmp_a = FETCH_FLOAT4(a[idx]);
        float4 tmp_b = FETCH_FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        FETCH_FLOAT4(c[idx]) = tmp_c;
    }
}
int block_size = 1024;
int grid_size  = CEIL(N/4, block_size);
elementwise_add_float4<<<grid_size, block_size>>>(a, b, c, N);

/*
    sigmod
*/
__global__ void sigmod(flaot* x,float* y,int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        y[idx] = 1.f/(1.0f + expf(-x[idx]));
    }
}

__global__ void sigmod_float4(flaot* x,float* y,int N){
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx < N){
        float4 tmp_x = FETCH_FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = 1.f/(1.0f + expf(tmp_x.x));;
        tmp_y.y = 1.f/(1.0f + expf(tmp_x.y));;
        tmp_y.z = 1.f/(1.0f + expf(tmp_x.z));;
        tmp_y.w = 1.f/(1.0f + expf(tmp_x.w));;
        FETCH_FLOAT4(y[idx]) = tmp_y;
    }
}

__global__ void relu(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}
// float4
__global__ void relu_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = fmaxf(0.0f, tmp_x.x);
        tmp_y.y = fmaxf(0.0f, tmp_x.y);
        tmp_y.z = fmaxf(0.0f, tmp_x.z);
        tmp_y.w = fmaxf(0.0f, tmp_x.w);
        FLOAT4(y[idx]) = tmp_y;
    }
}