#version 460 core

out vec4 FragColor;
in vec3 fragPos;
in vec3 fragNorm;

uniform vec4 deepColor;
uniform vec4 skyColor;
uniform vec3 lightDir;
uniform vec3 eyePos;

void main()
{
	vec3 lightdir=normalize(lightDir);
	vec3 eyedir=normalize(fragPos-eyePos);
	vec3 fragnorm=normalize(fragNorm);
	float diffuse=max(0.0f,dot(lightdir,fragnorm));
	float facing=1.0f-max(0.0f,dot((eyedir+lightdir)/2.0f,fragnorm));
	float frenel=pow(facing,5.0f);
	FragColor=deepColor*diffuse+frenel*skyColor;
	//FragColor=deepColor*diffuse;
	//FragColor=vec4(0.0f,0.0f,1.0f,1.0f);
}