#include<stdio.h>

int main(){
	cudaDeviceProp prop;
	//获取参数
	cudaGetDeviceProperties(&prop,0);
	printf("---GPU设备基本信息---\n");
	printf("名称：%s\n",prop.name);
	printf("时钟频率：%d\n",prop.clockRate);
	printf("版本号：%d.%d\n",prop.major,prop.minor);
	printf("总显存大小：%ld\n",prop.totalGlobalMem);
	printf("常量内存大小：%ld\n",prop.totalConstMem);
	printf("微处理器数量：%d\n",prop.multiProcessorCount);
	printf("每个线程块的共享内存大小：%ld\n",prop.sharedMemPerBlock);
	printf("每个线程块中的寄存器数量：%d\n",prop.regsPerBlock);
	printf("线程块最大包含的线程数：%d\n",prop.maxThreadsPerBlock);
	printf("线程块的最大维度：(%d %d %d)\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("网格的最大维度：(%d %d %d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf("纹理内存最大维度：(%d %d %d)\n",prop.maxTexture3D[0],prop.maxTexture3D[1],prop.maxTexture3D[2]);
}
