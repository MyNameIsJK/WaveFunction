#include "Vertex.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <iostream>
typedef unsigned int uint;
__global__ void updateHeightKernel(float* preHeight, float* curHeight, float* nextHeight, float3 mk, int2 meshSize)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	int idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < (meshSize.x - 1) && idxy < (meshSize.y - 1))
	{
		uint basicIndex = idxy * meshSize.x + idxx;
		if (idxx < 1 || idxy < 1)
			return;
		float mh = 0.0f;
		mh += mk.x * preHeight[basicIndex];
		mh += mk.y * curHeight[basicIndex];
		mh += mk.z * (curHeight[basicIndex + 1]+ curHeight[basicIndex - 1] + curHeight[basicIndex + meshSize.x] + curHeight[basicIndex - meshSize.x]);
				
		nextHeight[basicIndex] = mh;
	}
}
__global__ void updateVertexKernel(Vertex* vertexDev, int2 meshSize, float*preHeight, float* curHeight, float*nextHeight)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	int idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < (meshSize.x - 1) && idxy < (meshSize.y - 1))
	{
		if (idxx < 1 || idxy < 1)
			return;
		uint basicIndex = idxy * meshSize.x + idxx;
		preHeight[basicIndex] = curHeight[basicIndex];
		curHeight[basicIndex] = nextHeight[basicIndex];
		float3 xgrad = make_float3(2.0f / meshSize.x, curHeight[basicIndex + 1] - curHeight[basicIndex - 1], 0.0f);
		float3 zgrad = make_float3(0.0f, curHeight[basicIndex + meshSize.x] - curHeight[basicIndex - meshSize.x], 2.0f / meshSize.x);
		vertexDev[basicIndex].norm = normalize(cross(zgrad, xgrad));
		vertexDev[basicIndex].pos.y = curHeight[basicIndex];
	}
}

extern "C" void cudaVertexUpdate(float* preHeight, float* curHeight, float* nextHeight, float3 mk, int2 meshSize,
	Vertex * vertexDev, dim3 gridSize, dim3 blockSize, size_t numMapBytes)
{
	//float* height = new float[meshSize.x * meshSize.y];
	updateHeightKernel << <gridSize, blockSize >> > (preHeight, curHeight, nextHeight, mk, meshSize);
	//checkCudaErrors(cudaMemcpy(preHeight, curHeight, numMapBytes, cudaMemcpyDeviceToDevice));
	//checkCudaErrors(cudaMemcpy(curHeight, nextHeight, numMapBytes, cudaMemcpyDeviceToDevice));
	//cudaMemcpy(height, preHeight, numMapBytes, cudaMemcpyDeviceToHost);
	//std::cout << height[100 * meshSize.x + 100] << std::endl;
	//delete[]height;
	updateVertexKernel << <gridSize, blockSize >> > (vertexDev, meshSize, preHeight, curHeight, nextHeight);
}