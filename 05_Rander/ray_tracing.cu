#include<stdio.h>
#include<opencv2/opencv.hpp>
#include"helper_math.h"
#include"Sphere.cu"
#define SPHERES 20
#define THREADS 32
#define FRAMES 108
using namespace cv;

__constant__ Sphere s_cuda[SPHERES*sizeof(Sphere)];

__global__ void ray_tracing(uchar *img){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int t = blockIdx.z;
	int offset = x + y*blockDim.x*gridDim.x + t*blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	float fx = (float)x - gridDim.x*blockDim.x/2;
	float fy = (float)y - gridDim.y*blockDim.y/2;
	float2 origin = make_float2(fx,fy);
	float3 color = make_float3(0,0,0);
	float max_z = -INF;
	float3 c;
	for(int i=0;i<SPHERES;i++){
		float z = s_cuda[i].hit(origin,&c,t);
		if(z>max_z){
			color = c;
			max_z = z;
		}
	}
	img[offset*3+0] = (uchar)color.z;
	img[offset*3+1] = (uchar)color.y;
	img[offset*3+2] = (uchar)color.x;
}

int main(){
	Sphere s[SPHERES];
	Mat img(DIM,DIM,CV_8UC3);
	uchar *img_cuda;
	char name[10];
	for(int i=0;i<SPHERES;i++)
		s[i].init();
	cudaMalloc((void**)&img_cuda,DIM*DIM*3*FRAMES*sizeof(uchar));
	cudaMemcpyToSymbol(s_cuda,s,sizeof(Sphere)*SPHERES);
	dim3 GRID(DIM/THREADS,DIM/THREADS,FRAMES);
	dim3 BLOCK(THREADS,THREADS);
	ray_tracing<<<GRID,BLOCK>>>(img_cuda);
	for(int i=0;i<FRAMES;i++){
		uchar* t = img_cuda+DIM*DIM*3*i;
		cudaMemcpy(img.data,t,DIM*DIM*3*sizeof(uchar),cudaMemcpyDeviceToHost);
		sprintf(name,"./tracing/%d.bmp",i);
		imwrite(name,img);
	}
	cudaFree(img_cuda);
	return 0;
}
