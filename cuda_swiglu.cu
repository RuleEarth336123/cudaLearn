/*
GLU(x) = Linear(x) ⊗ σ(Linear(x))
Swish(x) = x⋅σ(x) = x * 1 / (1 + e-x)
swiGLU(x) = x1 ⊗ Swish(x2)

*/

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear1(x) * F.silu(self.linear2(x))

__global__ void swiglu_kernel(float* input1,float* input2,int N){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int warpid = threadIdx.x / warpSize;
    const int laneid = threadIdx.x % warpSize;


    
}