#include<cufft.h>
#include<cmath>
#include<opencv2/opencv.hpp>
#define N 256
#define THREAD 16
using namespace cv;

__global__ void lowPass(cufftComplex *data, float d0){
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	uint offset = x*N + y;
	float dx = fabs((float)x - N/2);
	float dy = fabs((float)y - N/2);
	float d = sqrt(dx*dx + dy*dy);
	float filter;
	//ILPF
	//filter = d<=d0?1:0;
	//GLPF
	//filter = exp(-d*d/2/d0/d0);
	//BLPF
	filter = 1/(1+(d/d0)*(d/d0));
	data[offset].x *= filter;
	data[offset].y *= filter;
}

int main(){
	Mat img = imread("fft/test.jpg", IMREAD_GRAYSCALE);
	img.convertTo(img, CV_32F);
	Mat dft(N,N,CV_32F);
	cufftComplex* data;
	cufftHandle plan;
	cudaMallocManaged((void **)&data, N*N*sizeof(cufftComplex));
	cufftPlan2d(&plan, N, N, CUFFT_C2C);
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			float shift = (i+j)%2==0?1:-1;
			data[i*N+j].x = img.at<float>(i,j) * shift;
		}
	}
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	dim3 grid(N/THREAD,N/THREAD);
	dim3 block(THREAD,THREAD);
	lowPass<<<grid,block>>>(data, 20);
	cufftExecC2C(plan, data, data, CUFFT_INVERSE);
	cudaDeviceSynchronize();
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			float shift = (i+j)%2==0?1:-1;
			dft.at<float>(i,j) = data[i*N+j].x / (N*N) * shift;
		}
	}
	imwrite("fft/dft.jpg", dft);
	cudaFree(data);
	cufftDestroy(plan);
	return 0;
}
