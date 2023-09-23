#include<stdio.h>
const int N = 100;
const int BLOCK = 128;

__device__ int sum(int *cache, int id){
	int i = blockDim.x/2;
	while(i!=0){
		if(id < i){
			cache[id] += cache[id+i];
		}
		__syncthreads();
		i /= 2;
	}
	return cache[0];
}

__global__ void dot(int *a,int *b,int *c){
	__shared__ int cache[BLOCK];
	int t = threadIdx.x;
	if(t < N)
		cache[t] = a[t]*b[t];
	else
		cache[t] = 0;
	__syncthreads();
	sum(cache,t);
	__syncthreads();
	*c = cache[0];
}

__host__ int dot(int *a,int *b){
	int *a_cuda,*b_cuda,*c_cuda;
	int r;
	cudaMalloc((void**)&a_cuda,N*sizeof(int));
    cudaMalloc((void**)&b_cuda,N*sizeof(int));
    cudaMalloc((void**)&c_cuda,sizeof(int));
    cudaMemcpy(a_cuda,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda,b,N*sizeof(int),cudaMemcpyHostToDevice);
	dot<<<1,BLOCK>>>(a_cuda,b_cuda,c_cuda);
	cudaMemcpy(&r,c_cuda,sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
	cudaFree(c_cuda);
	return r;
}

int main(){
    int a[N],b[N],c;
    for(int i=0;i<N;i++){
    	a[i] = i;
    	b[i] = 1;
    }
    c = dot(a,b);
	printf("dot(a,b)=%d\n",c);
	return 0;
}
