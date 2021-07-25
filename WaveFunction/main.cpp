#include "OceanWave.h"
#include <helper_math.h>
#include <time.h>
typedef unsigned int uint;
using namespace std;

clock_t preClock;
clock_t curClock;
float oceanDisturbTime = 3.0f;
int2 oceanDisturbPos;
void CalculateFPS(OceanWave&ocean,GLFWwindow*window, int2&meshSize)
{
	static uint durationTime = 0;
	static int fps = 0;
	static int oceanTime = 0;
	durationTime += (curClock - preClock);
	oceanTime += (curClock - preClock);
	float durationTimef = (float)durationTime / (float)CLOCKS_PER_SEC;
	float oceanTimef = (float)oceanTime / (float)CLOCKS_PER_SEC;
	fps++;
	if (oceanTimef > oceanDisturbTime)
	{
		int randX, randY;
		float randM;
		randX = rand() % (meshSize.x - 600);
		randX += 300;
		randY = rand() % (meshSize.y - 600);
		randY += 300;
		int tmpRand = rand() % 3;
		tmpRand++;
		randM = (float)tmpRand / 200.0f;
		ocean.disturb(randX, randY, randM);
		oceanTime = 0;
	}
	if (durationTimef > 0.1f)
	{
		ostringstream oss;
		oss << "WaveFunc FPS: ";
		float fpsf = (float)fps / durationTimef;
		char curtime[100];
		sprintf(curtime, "%.2f", fpsf);
		oss << curtime;
		glfwSetWindowTitle(window, oss.str().c_str());
		fps = 0;
		durationTime = 0;
	}
}
int main()
{
	int winWidth = 800;
	int winHeight = 600;
	string winName = "Ocean";
	MyOpenGL myGL;
	Shader* myShader;
	GLFWwindow* window;
	uint VAO, VBO, EBO;
	myGL.initGLFW();
	window = myGL.createWindow(winWidth, winHeight, winName);
	myGL.initGLAD();
	myShader = new Shader("./shader/Ocean.vert", "./shader/Ocean.Frag");
	uint meshSize = 1024;
	OceanWave ocean(make_int2(meshSize), 0.03f, 1.0f, 4.0f, 0.2f);
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	ocean.bufferVertexData(VAO, VBO);
	ocean.bufferIndexData(EBO, VAO);
	//draw(M, N, vertices);
	glEnable(GL_DEPTH_TEST);
	cout << "初始化成功" << endl;
	glm::vec3 translateVec;
	glm::vec3 rotateVec;
	glm::mat4 model;
	glm::mat4 E(1.0f);
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 proj = glm::perspective(glm::radians(90.0f), (float)winWidth / (float)winHeight, 0.1f, 10.0f);
	ocean.disturb(100, 100, 0.02f);

	myShader->use();
	myShader->setVec4("deepColor", glm::vec4(0.0f, 0.1f, 0.4f, 1.0f));
	myShader->setVec4("skyColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
	myShader->setVec3("lightDir", glm::vec3(0.0f, 1.0f, 0.0f));
	myShader->setVec3("eyePos", glm::vec3(0.0f, 0.0f, 5.0f));
	while (!glfwWindowShouldClose(window))
	{
		preClock = clock();
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//draw(M, N, vertices);
		ocean.update();
		ocean.bufferVertexData(VAO, VBO);
		translateVec = myGL.getTrasVec();
		rotateVec = myGL.getRotateVec();
		model = translate(E, translateVec);
		model = glm::rotate(model, glm::radians(rotateVec.x), glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, glm::radians(rotateVec.y), glm::vec3(0.0f, 1.0f, 0.0f));
		myShader->use();
		myShader->setMat4("transform", proj * view * model);
		glDrawElements(GL_TRIANGLES, (meshSize - 1) * (meshSize - 1) * 6, GL_UNSIGNED_INT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
		curClock = clock();
		CalculateFPS(ocean, window, make_int2(meshSize));
	}
	glfwTerminate();
	return 0;
}