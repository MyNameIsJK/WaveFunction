#pragma once
#ifndef OCEAN_H
#define OCEAN_H
#include "Vertex.h"
#include "MyOpenGL.h"
#include <helper_cuda.h>
typedef unsigned int uint;
class OceanWave
{
private:
	Vertex* vertices;
	uint* indices;
	float* curHeight;
	int2 oceanMeshSize;

	uint numVertex;
	size_t numVertBytes;
	size_t numMapBytes;

	dim3 gridSize;
	dim3 blockSize;

	float3 mk;
	float dt;

	// device data
	float* preHeightDev;
	float* curHeightDev;
	float* nextHeightDev;
	Vertex* vertDev;
public:
	OceanWave(int2 oceanMeshSize, float dt, float dx, float speed, float damping);
	~OceanWave();
	OceanWave() = delete;
	OceanWave(const OceanWave& ow) = delete;
	OceanWave& operator = (const OceanWave& ow) = delete;

	Vertex* getVertices();
	uint* getIndices();
	void bufferVertexData(uint& VAO, uint& VBO);
	void bufferIndexData(uint& EBO, uint& VAO);
	void disturb(int x, int y, float magnitude);
	void update();
};
#endif
