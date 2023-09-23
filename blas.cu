#include<cstdio>
#include<vector>
#include<cuda_runtime.h>
#include<cublas_v2.h>
using namespace std;

int main(){
    /* --- step 1. create handle --- */
	cublasHandle_t handle = NULL;
	cublasCreate(&handle);

    /* --- step 2. copy data to GPU --- */
	vector<float> a = {1,1,0,1};
	vector<float> b = {1,2,3,4};
	vector<float> c(4);
	const float alpha = 1;
	const float beta = 0;
	float *a_dev = nullptr;
	float *b_dev = nullptr;
	float *c_dev = nullptr;
	cudaMalloc((void **)&a_dev, a.size() * sizeof(float));
	cudaMalloc((void **)&b_dev, b.size() * sizeof(float));
	cudaMalloc((void **)&c_dev, c.size() * sizeof(float));
	cudaMemcpy(a_dev, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);

    /* --- step 3. compute by cuBLAS functions --- */
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 2, 2, 2, &alpha, a_dev, 2, b_dev, 2, &beta, c_dev, 2);
	
    /* --- step 4. return data to CPU and show ---*/
	cudaMemcpy(c.data(), c_dev, c.size()*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i<c.size();i++)
		printf("%f ",c[i]);
	printf("\n");

    /* --- step 5.free memory ---*/
	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);
	cublasDestroy(handle);
	return 0;
}
