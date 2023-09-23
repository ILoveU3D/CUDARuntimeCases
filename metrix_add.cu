#include<stdio.h>
#define DIMX 10
#define DIMY 10

__global__ void add(int *a,int *b,int *c){
	int x=blockIdx.x;
	int y=blockIdx.y;
	int offset=x*gridDim.y+y;
	if(x<DIMX&&y<DIMY)
		c[offset]=a[offset]+b[offset];
}

int main(){
	int a[DIMX*DIMY],b[DIMX*DIMY],c[DIMX*DIMY];
	int *a_cuda,*b_cuda,*c_cuda;
	//赋值
	for(int i=0;i<DIMX;i++){
		for(int j=0;j<DIMY;j++){
			a[i*DIMX+j]=i+j-3;
			b[i*DIMX+j]=(i+j)/2+1;
		}
	}
	cudaMalloc((void**)&a_cuda,DIMX*DIMY*sizeof(int));
	cudaMalloc((void**)&b_cuda,DIMX*DIMY*sizeof(int));
	cudaMalloc((void**)&c_cuda,DIMX*DIMY*sizeof(int));
	cudaMemcpy(a_cuda,a,DIMX*DIMY*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(b_cuda,b,DIMX*DIMY*sizeof(int),cudaMemcpyHostToDevice);
	dim3 grid(DIMX,DIMY);
	add<<<grid,1>>>(a_cuda,b_cuda,c_cuda);
	cudaMemcpy(c,c_cuda,DIMX*DIMY*sizeof(int),cudaMemcpyDeviceToHost);
	printf("a+b=(");
	for(int i=0;i<DIMX;i++){
		for(int j=0;j<DIMY;j++){
			printf("%d,",c[i*DIMX+j]);
		}
		printf("\n");
	}
	printf(")\n");
	cudaFree(a_cuda);
	cudaFree(b_cuda);
	cudaFree(c_cuda);
}
