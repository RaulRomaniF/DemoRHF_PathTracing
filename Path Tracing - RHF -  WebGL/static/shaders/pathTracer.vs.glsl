#version 300 es 

precision highp float;

layout(location = 0) in vec2 vPos;
out vec2 vUV;
 
void main()
{  
	//set the current object space position as the output texture coordinates and the clip space position
	vUV = vPos;	
	gl_Position = vec4(vPos.xy ,0 ,1);
}