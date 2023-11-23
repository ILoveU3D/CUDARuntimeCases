#include<stdio.h>
#include<math.h>
#include<curand_kernel.h>
#define THREADS 16
#define N 65536
#define XMAX 3.14
#define XMIN 0
#define YMAX 3.14
#define YMIN 0

__device__ float function(float x, float y){
	return y*sin(x);
}

__global__ void monte_carlo(float* mark, curandState* state, unsigned long seed){
	uint t = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, t, 0, &state[t]);
	float x = curand_uniform(&state[t]) * (XMAX-XMIN) + XMIN;
	float y = curand_uniform(&state[t]) * (YMAX-YMIN) + YMIN;
	float f = function(x,y);
	atomicAdd(mark, f * (XMAX-XMIN) * (YMAX-YMIN));
}

int main(){
	curandState* state;
	float *mark;
	cudaMalloc(&state, N*sizeof(curandState));
	cudaMallocManaged((void **)&mark, sizeof(float));
	*mark = 0;
	monte_carlo<<<N/THREADS,THREADS>>>(mark, state, time(NULL));
	cudaDeviceSynchronize();
	printf("int is:%f\n", *mark/N);
	cudaFree(mark);
	cudaFree(state);
	return 0;
}