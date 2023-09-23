#include<stdio.h>
#include<time.h>
#define N 10*65536

__global__ void add(int *a,int *b,int *c){
	int t=blockIdx.x*blockDim.x+threadIdx.x;
	if(t<N)
		c[t]=a[t]+b[t];
}

int main(){
	int a[N],b[N],c[N];
	int *a_cuda,*b_cuda,*c_cuda;
	//赋值
	for(int i=0;i<N;i++){
		a[i]=i-3;
		b[i]=i/2+1;
	}
	time_t start,end;
	start = time(NULL);
	cudaMalloc((void**)&a_cuda,N*sizeof(int));
	cudaMalloc((void**)&b_cuda,N*sizeof(int));
	cudaMalloc((void**)&c_cuda,N*sizeof(int));
	cudaMemcpy(a_cuda,a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(b_cuda,b,N*sizeof(int),cudaMemcpyHostToDevice);
	add<<<N/128,128>>>(a_cuda,b_cuda,c_cuda);
	cudaMemcpy(c,c_cuda,N*sizeof(int),cudaMemcpyDeviceToHost);
	end = time(NULL);
	printf("time=%fs\n",difftime(end,start));
	cudaFree(a_cuda);
	cudaFree(b_cuda);
	cudaFree(c_cuda);
}
