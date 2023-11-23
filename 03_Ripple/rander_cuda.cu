#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<math.h>
#define WIDTH 1024
#define HEIGHT 1024
#define FRAMES 108
using namespace cv;

__global__ void generate(uchar *data){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int t = blockIdx.z;
	int offset = x + y*blockDim.x*gridDim.x + t*blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	float fx = (float)x - gridDim.x*blockDim.x/2;
	float fy = (float)y - gridDim.y*blockDim.y/2;
	float d = sqrt(fx*fx+fy*fy);
	float mag = 10.0f;
	float value = 128.0f + 127.0f*cos((d-t)/mag)/(d/mag+1.0f);
	data[offset] = (uchar)value;
}

int main(){
	uchar *data_cuda;
	Mat img(WIDTH, HEIGHT, CV_8U);
	char name[10];
	cudaMalloc((void**)&data_cuda,FRAMES*WIDTH*HEIGHT*sizeof(uchar));
	int threadPerBlock = 32;
	dim3 grid(WIDTH/threadPerBlock,HEIGHT/threadPerBlock,FRAMES);
	dim3 block(threadPerBlock,threadPerBlock);
	generate<<<grid,block>>>(data_cuda);
	for(int i=0;i<FRAMES;i++){
		uchar* t = data_cuda+WIDTH*HEIGHT*i;
		cudaMemcpy(img.data,t,WIDTH*HEIGHT*sizeof(uchar),cudaMemcpyDeviceToHost);
		sprintf(name,"./frames/%d.bmp",i);
		imwrite(name,img);
	}
	return 0;
}
