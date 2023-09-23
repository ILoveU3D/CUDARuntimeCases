#include<opencv2/opencv.hpp>
#include<math.h>
#include<stdio.h>
#define W 512
#define H 512
#define THREADS 32
using namespace cv;

__device__ float normalize(float *u,float *d, int offset){
	return d[offset]!=0.0 ? u[offset]/d[offset] : 0;
}

__global__ void cal_uxy(float *u,float *u_x, float *u_y){
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	if(x+1<W)
		u_x[x*W+y] = u[(x+1)*W+y] - u[x*W+y];
	else
		u_x[x*W+y] = u[x*W+y];
	if(y+1<H)
		u_y[x*W+y] = u[x*W+y+1] - u[x*W+y];
	else
		u_y[x*W+y] = u[x*W+y];
}

__global__ void cal_ud(float *u_x,float *u_y,float *u_d){
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	u_d[x*W+y] = sqrt(u_x[x*W+y]*u_x[x*W+y]+u_y[x*W+y]*u_y[x*W+y]);
}

__global__ void cal_uxy1(float *u_x,float *u_y,float *u_d,float *u_x1,float *u_y1){
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	if(x-1>=0)
		u_x1[x*W+y] = normalize(u_x,u_d,x*W+y) - normalize(u_x,u_d,(x-1)*W+y);
	else
		u_x1[x*W+y] = normalize(u_x,u_d,x*W+y);
	if(y-1>=0)
		u_y1[x*W+y] = normalize(u_y,u_d,x*W+y) - normalize(u_y,u_d,x*W+y-1);
	else
		u_y1[x*W+y] = normalize(u_y,u_d,x*W+y);
}

__global__ void cal_new(float *u,float *u0,float *u_x1,float *u_y1,float *u_temp, const float *lambda, const float *t){
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	float grad = *lambda * (u[x*W+y]-u0[x*W+y]) - u_x1[x*W+y] - u_y1[x*W+y];
	u_temp[x*W+y] = u[x*W+y] - *t * grad;
}

int main(){
	Mat img = imread("./tv/demo.jpeg",IMREAD_GRAYSCALE);
	img.convertTo(img, CV_32F);
	Mat result(W,H,CV_32F);
	const float t = 1.0, lambda = 0.01;
	int epoch = 20;
	float *u0,*u,*u_temp;
	float *u_x,*u_y,*u_d,*u_x1,*u_y1,*lambda_cuda,*t_cuda;
	cudaMalloc((void**)&u0,W*H*sizeof(float));
	cudaMalloc((void**)&u_x,W*H*sizeof(float));
	cudaMalloc((void**)&u_y,W*H*sizeof(float));
	cudaMalloc((void**)&u,W*H*sizeof(float));
	cudaMalloc((void**)&u_temp,W*H*sizeof(float));
	cudaMalloc((void**)&u_d,W*H*sizeof(float));
	cudaMalloc((void**)&u_x1,W*H*sizeof(float));
	cudaMalloc((void**)&u_y1,W*H*sizeof(float));
	cudaMalloc((void**)&lambda_cuda,sizeof(float));
	cudaMalloc((void**)&t_cuda,sizeof(float));
	cudaMemcpy(u0, (float*)img.data, W*H*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(u, (float*)img.data, W*H*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(t_cuda, &t, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lambda_cuda, &lambda, sizeof(float), cudaMemcpyHostToDevice);
	dim3 GRID(W/THREADS,H/THREADS);
	dim3 BLOCK(THREADS,THREADS);
	for(int i=0;i<epoch;i++){
		printf("iterate %d ...\n",i);
		cal_uxy<<<GRID,BLOCK>>>(u,u_x,u_y);
		cal_ud<<<GRID,BLOCK>>>(u_x,u_y,u_d);
		cal_uxy1<<<GRID,BLOCK>>>(u_x,u_y,u_d,u_x1,u_y1);
		cal_new<<<GRID,BLOCK>>>(u,u0,u_x1,u_y1,u_temp,lambda_cuda,t_cuda);
		cudaMemcpy(u, u_temp, W*H*sizeof(float),cudaMemcpyDeviceToDevice);
	}
	cudaMemcpy((float*)result.data, u, W*H*sizeof(float),cudaMemcpyDeviceToHost);
	imwrite("./tv/result.bmp",result);
	cudaFree(u0);
	cudaFree(u);
	cudaFree(u_temp);
	cudaFree(u_x);
	cudaFree(u_y);
	cudaFree(u_d);
	cudaFree(u_x1);
	cudaFree(u_y1);
	cudaFree(lambda_cuda);
	cudaFree(t_cuda);
	return 0;
}
