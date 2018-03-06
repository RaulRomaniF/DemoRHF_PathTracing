#version 330 core
// #version 300 es
// precision mediump float;

layout (location=0) out vec4 vFragColor; //fragment shader output

void main()
{
	//returns a constant colour (white)
	vFragColor = vec4(1,1,1,1);
}