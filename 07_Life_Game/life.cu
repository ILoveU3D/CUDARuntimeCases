#include<opencv2/opencv.hpp>
#include<stdio.h>
#define N 512
#define THREADS 32
#define LIFE 0
#define DEATH 255
using namespace cv;

texture<uchar, 2> map;

__device__ inline bool alive(uchar pixel){
	return pixel > uchar(128)? 0:1;
}

__global__ void evolute(uchar *after){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	uchar value = tex2D(map,x,y);
	int sum = 0;
	bool north = alive(tex2D(map,x,y-1));
	sum += north;
	bool south = alive(tex2D(map,x,y+1));
	sum += south;
	bool east = alive(tex2D(map,x+1,y));
	sum += east;
	bool west = alive(tex2D(map,x-1,y));
	sum += west;
	bool northeast = alive(tex2D(map,x+1,y-1));
	sum += northeast;
	bool northwest = alive(tex2D(map,x-1,y-1));
	sum += northwest;
	bool southeast = alive(tex2D(map,x+1,y+1));
	sum += southeast;
	bool southwest = alive(tex2D(map,x-1,y+1));
	sum += southwest;
	if(alive(value)){
		if(sum<2||sum>5)
			value = DEATH;
		else if(north&&south&&east&&west)
			value = DEATH;
		else
			value = LIFE;
	}else{
		if(sum==4)
			value = LIFE;
		else if(north&&south||east&&west)
			value = LIFE;
		else
			value = DEATH;
	}
	after[x*N+y] = value;
}

int main(){
	Mat img = imread("init.png",IMREAD_GRAYSCALE);
	uchar *map_cuda,*t;
	int times = 100;
	char name[20];
	cudaMalloc((void**)&map_cuda, N*N*sizeof(uchar));
	cudaMalloc((void**)&t, N*N*sizeof(uchar));
	cudaMemcpy(map_cuda, img.data, N*N*sizeof(uchar), cudaMemcpyHostToDevice);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
	cudaBindTexture2D(NULL, map, map_cuda, desc, N, N, sizeof(uchar)*N);
	dim3 GRID(N/THREADS,N/THREADS);
	dim3 BLOCK(THREADS,THREADS);
	for(int i=0;i<times;i++){
		evolute<<<GRID,BLOCK>>>(t);
		cudaMemcpy(img.data, map_cuda, N*N*sizeof(uchar), cudaMemcpyDeviceToHost);
		sprintf(name, "./lifegame/%d.png", i);
		imwrite(name, img);
		cudaMemcpy(map_cuda,t,N*N*sizeof(uchar),cudaMemcpyDeviceToDevice);
	}
	cudaUnbindTexture(map);
	cudaFree(map_cuda);
	cudaFree(t);
	return 0;
}
