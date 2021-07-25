#include "OceanWave.h"
extern "C" void createMesh(dim3 blockSize, dim3 gridSize, 
	int2 oceanMeshSize, uint * indicesHost, Vertex * vertHost, Vertex * vertDev);
extern "C" void cudaVertexUpdate(float* preHeight, float* curHeight, float* nextHeight, float3 mk, int2 meshSize,
	Vertex * vertexDev, dim3 gridSize, dim3 blockSize, size_t numMapBytes);
int iDivUp(int x, int y)
{
	int ret = x / y;
	if (x % y != 0)
		ret++;
	return ret;
}

//float* nextHeight;
OceanWave::OceanWave(int2 oceanMeshSize, float dt, float dx, float speed, float damping):
	oceanMeshSize(oceanMeshSize),dt(dt)
{
	numVertex = oceanMeshSize.x * oceanMeshSize.y;
	numVertBytes = numVertex * sizeof(Vertex);
	numMapBytes = numVertex * sizeof(float);
	vertices = new Vertex[numVertex];
	indices = new uint[(oceanMeshSize.x - 1) * (oceanMeshSize.y - 1) * 6];
	curHeight = new float[numVertex];
	//nextHeight = new float[numVertex];

	checkCudaErrors(cudaMalloc((void**)&vertDev, numVertBytes));
	checkCudaErrors(cudaMalloc((void**)&preHeightDev, numMapBytes));
	checkCudaErrors(cudaMalloc((void**)&curHeightDev, numMapBytes));
	checkCudaErrors(cudaMalloc((void**)&nextHeightDev, numMapBytes));

	checkCudaErrors(cudaMemset(preHeightDev, 0, numMapBytes));
	checkCudaErrors(cudaMemset(curHeightDev, 0, numMapBytes));
	checkCudaErrors(cudaMemset(nextHeightDev, 0, numMapBytes));

	blockSize = dim3(16, 16);
	gridSize = dim3(iDivUp(oceanMeshSize.x, 16), iDivUp(oceanMeshSize.y, 16));
	createMesh(blockSize, gridSize, oceanMeshSize, indices, vertices, vertDev);
	memset(curHeight, 0, numMapBytes);

	float d = damping * dt + 2.0f;
	float ed = speed * speed * dt * dt / dx / dx;
	mk.x = (damping * dt - 2.0f) / d;
	mk.y = (4.0f - 8.0f * ed) / d;
	mk.z = 2.0f * ed / d;

}

OceanWave::~OceanWave()
{
	delete[]curHeight;
	delete[]vertices;
	delete[]indices;
	cudaFree(vertDev);
	cudaFree(preHeightDev);
	cudaFree(curHeightDev);
	cudaFree(nextHeightDev);
}

Vertex* OceanWave::getVertices()
{
	return vertices;
}

uint* OceanWave::getIndices()
{
	return indices;
}

void OceanWave::bufferVertexData(uint& VAO, uint& VBO)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, numVertBytes, vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(float3)));
}

void OceanWave::bufferIndexData(uint& EBO, uint& VAO)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (oceanMeshSize.x - 1) * (oceanMeshSize.y - 1) * 6 * sizeof(uint), indices, GL_STATIC_DRAW);
}

void OceanWave::disturb(int x, int y, float magnitude)
{
	if (x > 1 && x < (oceanMeshSize.x - 1) && y>1 && y < (oceanMeshSize.y - 1))
	{
		float halfMag = 0.5f * magnitude;
		int index = y * oceanMeshSize.x + x;
		/*
		vertices[index].pos.y += magnitude;
		vertices[index - 1].pos.y += halfMag;
		vertices[index + 1].pos.y += halfMag;
		vertices[index - oceanMeshSize.x].pos.y += halfMag;
		vertices[index + oceanMeshSize.x].pos.y += halfMag;
		*/
		curHeight[index] += magnitude;
		curHeight[index + 1] += halfMag;
		curHeight[index - 1] += halfMag;
		curHeight[index + oceanMeshSize.x] += halfMag;
		curHeight[index - oceanMeshSize.x] += halfMag;
		checkCudaErrors(cudaMemcpy(preHeightDev, curHeightDev, numMapBytes, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(curHeightDev, curHeight, numMapBytes, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(curHeight, curHeightDev, numMapBytes, cudaMemcpyDeviceToHost));
		//cout << "3 " << curHeight[100 * oceanMeshSize.x + 100] << endl;
		
	}
}
void OceanWave::update()
{
	//cout << "1 " << curHeight[100 * oceanMeshSize.x + 100] << endl;
	
	cudaVertexUpdate(preHeightDev, curHeightDev, nextHeightDev, mk, oceanMeshSize,
		vertDev, gridSize, blockSize, numMapBytes);
	checkCudaErrors(cudaMemcpy(curHeight, curHeightDev, numMapBytes, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vertices, vertDev, numVertBytes, cudaMemcpyDeviceToHost));
	//float maxh = -1.0f;
	//int mi, mj;
	//for (int i = 0; i < oceanMeshSize.x; i++)
	//	for (int j = 0; j < oceanMeshSize.y; j++)
	//		if (curHeight[j * oceanMeshSize.x + i] > maxh)
	//		{
	//			maxh = curHeight[j * oceanMeshSize.x + i];
	//			mi = i; mj = j;
	//		}
	//cout << mi << " " << mj <<" "<<maxh<< endl;
	//for (int i = 0; i < oceanMeshSize.x; i++)
	//	for (int j = 0; j < oceanMeshSize.y; j++)
	//		vertices[j * oceanMeshSize.x + i].pos.y = curHeight[j * oceanMeshSize.x + i];
	//cout << "2 " << curHeight[100 * oceanMeshSize.x + 100] << endl;
	/*
	int width = oceanMeshSize.x;
	int height = oceanMeshSize.y;
	for (int i = 1; i < width - 1; i++)
		for (int j = 1; j < height - 1; j++)
		{
			float mh = mk.x * curHeight[j * width + i] +
				mk.y * vertices[j * width + i].pos.y +
				mk.z * (vertices[j * width + i + 1].pos.y +
					vertices[j * width + i - 1].pos.y +
					vertices[j * width + i + width].pos.y +
					vertices[j * width + i - width].pos.y);
			nextHeight[j * width + i] = mh;
		}
	for (int i = 1; i < width - 1; i++)
		for (int j = 1; j < height - 1; j++)
		{
			curHeight[j * width + i] = vertices[j * width + i].pos.y;
			vertices[j * width + i].pos.y = nextHeight[j * width + i];
		}
	*/
}
