
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

// #include "..\src\GLSLShader.h"
#include "src/GLSLShader.h"

#include <vector>
#include "Object.h"
#include "model.h"

#include <SOIL.h>

#include "bitmap_image.hpp"

#define GL_CHECK_ERRORS assert(glGetError()== GL_NO_ERROR);

// #ifdef _DEBUG
// #pragma comment(lib, "glew_static_x86_d.lib")
// #pragma comment(lib, "freeglut_static_x86_d.lib")
// #pragma comment(lib, "SOIL_static_x86_d.lib")
// #else
// #pragma comment(lib, "glew_static_x86.lib")
// #pragma comment(lib, "freeglut_static_x86.lib")
// #pragma comment(lib, "SOIL_static_x86.lib")
// #endif

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);


using namespace std;
using namespace cv;

//output screen resolution
GLFWwindow* window;
const int WIDTH  = 800;
const int HEIGHT = 600;


bool keys[1024];
GLfloat lastX = 400, lastY = 300;
bool firstMouse = true;

GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;




//shaders for use in the recipe
//mesh rendering shader, pathtracing shader and flat shader
GLSLShader shader, pathtraceShader, flatShader;

//IDs for vertex array and buffer object
GLuint vaoID;
GLuint vboVerticesID;
GLuint vboIndicesID;

//projection and modelview matrices
glm::mat4  P = glm::mat4(1);
glm::mat4 MV = glm::mat4(1);

//Objloader instance
ObjLoader obj;
vector<Mesh*> meshes;				//all meshes
vector<Material_> materials;		//all materials
vector<unsigned short> indices;		//all mesh indices
// vector<Vertex> vertices;			//all mesh vertices
vector<GLuint> textures;			//all textures

//camera transformation variables
int state = 0, oldX = 0, oldY = 0;
float rX = 22, rY = 116, dist = -120;

//OBJ mesh filename to load
// const std::string mesh_filename = "model/bloques3.obj";
const std::string mesh_filename = "model/blocks.obj";

//flag to enable raytracing
bool bPathtrace = true;
// bool bPathtrace = false;

//fullscreen quad vao and vbos
GLuint quadVAOID;
GLuint quadVBOID;
GLuint quadIndicesID;

//background color
glm::vec4 bg = glm::vec4(52.9 / 255, 80.8 / 255, 92.2 / 255, 1);


glm::vec3 eyePos;
BBox aabb;

GLuint texVerticesID; //texture storing vertex positions
GLuint texTrianglesID; //texture storing triangles list

//light crosshair gizmo vetex array and buffer object IDs
GLuint lightVAOID;
GLuint lightVerticesVBO;
glm::vec3 lightPosOS = glm::vec3(0, 2, 0); //objectspace light position

//spherical cooridate variables for light rotation
float theta = 0.66f;
float phi = -1.0f;
float radius = 70;

//FPS related variables
int total_frames = 0;
float fps = 0;
float lastTime = 0;

//texture ID for array texture
GLuint textureID;

//OpenGL initialization function
void initialize() {
	//setup fullscreen quad geometry
	glm::vec2 quadVerts[4];
	quadVerts[0] = glm::vec2(-1, -1);
	quadVerts[1] = glm::vec2( 1, -1);
	quadVerts[2] = glm::vec2( 1, 1);
	quadVerts[3] = glm::vec2(-1, 1);

	//setup quad indices
	GLushort quadIndices[] = { 0, 1, 2, 0, 2, 3};

	//setup quad vertex array and vertex buffer objects
	glGenVertexArrays(1, &quadVAOID);
	glGenBuffers(1, &quadVBOID);
	glGenBuffers(1, &quadIndicesID);

	glBindVertexArray(quadVAOID);
	glBindBuffer (GL_ARRAY_BUFFER, quadVBOID);
	//pass quad vertices to vertex buffer object
	glBufferData (GL_ARRAY_BUFFER, sizeof(quadVerts), &quadVerts[0], GL_STATIC_DRAW);

	GL_CHECK_ERRORS

	//enable vertex attribute array for vertex position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

	//pass quad indices to element array buffer
	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, quadIndicesID);
	glBufferData (GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), &quadIndices[0], GL_STATIC_DRAW);

	//get the mesh path for loading of textures
	std::string mesh_path = mesh_filename.substr(0, mesh_filename.find_last_of("/") + 1);

	//load the obj model
	vector<unsigned short> indices2;
	vector<glm::vec3> vertices2;


	Model model_cubes("model/bloques_triang_v3.obj");

	indices2 = model_cubes.vertexIndices;
	vertices2 = model_cubes.temp_vertices;
	aabb.min = model_cubes.box.min;
	aabb.max = model_cubes.box.max;

	for (int i = 0; i < model_cubes.materials.size(); ++i)
		materials.push_back(model_cubes.materials[i] ) ;


	//NEW    NEW        NEW
	// if(!obj.Load(mesh_filename.c_str(), materials, aabb, vertices2, indices2)) {
	// 	cout<<"Cannot load the 3ds mesh"<<endl;
	// 	exit(EXIT_FAILURE);
	// }

	GL_CHECK_ERRORS

	int total = 0;
	// check the total number of non empty textures since we will use this
	// information to creare a single array texture to store all textures
	for (size_t k = 0; k < materials.size(); k++) {
		if (materials[k].map_Kd != "") {
			total++;
		}
	}

	cout << "Material size: " << materials.size() << endl;

	//load material textures
	for (size_t k = 0; k < materials.size(); k++) {
		//if the diffuse texture name is not empty

		if (materials[k].map_Kd != "") {
			cout << "Material name: " << mesh_path + materials[k].map_Kd << endl;

			if ( k == 0 ) {
				//generate a new OpenGL array texture
				glGenTextures(1, &textureID);
				glBindTexture(GL_TEXTURE_2D_ARRAY, textureID);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}

			int texture_width = 0, texture_height = 0, channels = 0;

			const string& filename = materials[k].map_Kd;

			std::string full_filename = mesh_path;
			full_filename.append(filename);

			//use SOIL to load the texture
			// std::cout<<"full path name: "<<full_filename<<endl;
			GLubyte* pData = SOIL_load_image(full_filename.c_str(), &texture_width, &texture_height, &channels, SOIL_LOAD_AUTO);
			if (pData == NULL) {
				cerr << "Cannot load image: " << full_filename.c_str() << endl;
				exit(EXIT_FAILURE);
			}

			//Flip the image on Y axis
			// int i,j;
			// for( j = 0; j*2 < texture_height; ++j )
			// {
			// 	int index1 = j * texture_width * channels;
			// 	int index2 = (texture_height - 1 - j) * texture_width * channels;
			// 	for( i = texture_width * channels; i > 0; --i )
			// 	{
			// 		GLubyte temp = pData[index1];
			// 		pData[index1] = pData[index2];
			// 		pData[index2] = temp;
			// 		++index1;
			// 		++index2;
			// 	}
			// }
			//get the image format
			GLenum format = GL_RGBA;
			switch (channels) {
			case 2:	format = GL_RG32UI; break;
			case 3: format = GL_RGB;	break;
			case 4: format = GL_RGBA;	break;
			}

			//if this is the first texture, allocate the array texture
			if (k == 0) {
				glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, format, texture_width, texture_height, total, 0, format, GL_UNSIGNED_BYTE, NULL);
			}

			//modify the existing texture            GLint zoffset
			glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, k, texture_width, texture_height, 1, format, GL_UNSIGNED_BYTE, pData);

			//release the SOIL image data
			SOIL_free_image_data(pData);
		}
	}



	// GL_CHECK_ERRORS

	//load flat shader
	flatShader.LoadFromFile(GL_VERTEX_SHADER, "shaders/flat.vert");
	flatShader.LoadFromFile(GL_FRAGMENT_SHADER, "shaders/flat.frag");
	//compile and link shader
	flatShader.CreateAndLinkProgram();
	flatShader.Use();
	//add attribute and uniform
	flatShader.AddAttribute("vVertex");
	flatShader.AddUniform("MVP");
	flatShader.UnUse();

	//load pathtracing shader
	pathtraceShader.LoadFromFile(GL_VERTEX_SHADER, "shaders/pathtracer.vert");
	pathtraceShader.LoadFromFile(GL_FRAGMENT_SHADER, "shaders/pathtracer.frag");
	//compile and link shader
	pathtraceShader.CreateAndLinkProgram();
	pathtraceShader.Use();
	//add attribute and uniform
	pathtraceShader.AddAttribute("vVertex");
	pathtraceShader.AddUniform("eyePos");
	pathtraceShader.AddUniform("invMVP");
	pathtraceShader.AddUniform("light_position");
	pathtraceShader.AddUniform("backgroundColor");
	pathtraceShader.AddUniform("aabb.min");
	pathtraceShader.AddUniform("aabb.max");
	pathtraceShader.AddUniform("vertex_positions");
	pathtraceShader.AddUniform("triangles_list");
	pathtraceShader.AddUniform("time");
	pathtraceShader.AddUniform("s");
	pathtraceShader.AddUniform("spp");
	pathtraceShader.AddUniform("VERTEX_TEXTURE_SIZE");
	pathtraceShader.AddUniform("TRIANGLE_TEXTURE_SIZE");

	//set values of constant uniforms as initialization
	glUniform1f(pathtraceShader("VERTEX_TEXTURE_SIZE"), (float)vertices2.size());
	glUniform1f(pathtraceShader("TRIANGLE_TEXTURE_SIZE"), (float)indices2.size() / 4);

	glUniform3fv(pathtraceShader("aabb.min"), 1, glm::value_ptr(aabb.min));
	glUniform3fv(pathtraceShader("aabb.max"), 1, glm::value_ptr(aabb.max));
	glUniform4fv(pathtraceShader("backgroundColor"), 1, glm::value_ptr(bg));
	glUniform1i( pathtraceShader("vertex_positions"), 1);
	glUniform1i( pathtraceShader("triangles_list")  , 2);

	cout << "VERTEX_TEXTURE_SIZE: "  << (float)vertices2.size() << endl;
	cout << "TRIANGLE_TEXTURE_SIZE: " << (float)indices2.size() / 4 << endl;
	cout << "aabb.min: " << aabb.min.x << ", " << aabb.min.y << ", " << aabb.min.z << endl;
	cout << "aabb.max: " << aabb.max.x << ", " << aabb.max.y << ", " << aabb.max.z << endl;
	pathtraceShader.UnUse();

	// GL_CHECK_ERRORS
	// check OpenGL error
	GLenum err;
	cout << "Start printing erros:" << endl;
	while ((err = glGetError()) != GL_NO_ERROR) {
		cerr << "OpenGL error: " << err << endl;
	}
	cout << "All errors are printed" << endl;


	//setup vao and vbo stuff for the light position crosshair
	glm::vec3 crossHairVertices[6];
	crossHairVertices[0] = glm::vec3(-0.5f, 0, 0);
	crossHairVertices[1] = glm::vec3(0.5f, 0, 0);
	crossHairVertices[2] = glm::vec3(0, -0.5f, 0);
	crossHairVertices[3] = glm::vec3(0, 0.5f, 0);
	crossHairVertices[4] = glm::vec3(0, 0, -0.5f);
	crossHairVertices[5] = glm::vec3(0, 0, 0.5f);

	//setup light gizmo vertex array and vertex buffer object IDs
	glGenVertexArrays(1, &lightVAOID);
	glGenBuffers(1, &lightVerticesVBO);
	glBindVertexArray(lightVAOID);

	glBindBuffer (GL_ARRAY_BUFFER, lightVerticesVBO);
	//pass crosshair vertices to the buffer object
	glBufferData (GL_ARRAY_BUFFER, sizeof(crossHairVertices), &(crossHairVertices[0].x), GL_STATIC_DRAW);
	GL_CHECK_ERRORS
	//enable vertex attribute array for vertex position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	GL_CHECK_ERRORS

	//use spherical coordinates to get the light position
	lightPosOS.x = radius * cos(theta) * sin(phi);
	lightPosOS.y = radius * cos(phi);
	lightPosOS.z = radius * sin(theta) * sin(phi);








	//pass position to 1D texture bound to texture unit 1
	glGenTextures(1, &texVerticesID);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture( GL_TEXTURE_2D, texVerticesID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	cout << "VERTICES: " << vertices2.size() << endl;
	GLfloat* pData = new GLfloat[vertices2.size() * 4];
	int count = 0;
	for (size_t i = 0; i < vertices2.size(); i++) {
		pData[count++] = vertices2[i].x;
		pData[count++] = vertices2[i].y;
		pData[count++] = vertices2[i].z;
		pData[count++] = 0;
	}
	//allocate a floating point texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, vertices2.size(), 1, 0, GL_RGBA, GL_FLOAT, pData);

	//delete the data pointer
	delete [] pData;

	// GL_CHECK_ERRORS

	//store the mesh topology in another texture bound to texture unit 2
	glGenTextures(1, &texTrianglesID);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture( GL_TEXTURE_2D, texTrianglesID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	cout << "TRIANGLES: " << indices2.size() << endl;
	GLushort* pData2 = new GLushort[indices2.size()];
	count = 0;
	for (size_t i = 0; i < indices2.size(); i += 4) {
		pData2[count++] = (indices2[i + 0]);
		pData2[count++] = (indices2[i + 1]);
		pData2[count++] = (indices2[i + 2]);
		pData2[count++] = (indices2[i + 3]);
	}
	//allocate an integer format texture          width         height
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16I, indices2.size() / 4, 1, 0, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, pData2);

	//delete heap allocated buffer
	delete [] pData2;

	// GL_CHECK_ERRORS

	//set texture unit 0 as active texture unit
	glActiveTexture(GL_TEXTURE0);

	//enable depth test and culling
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	//set the background colour
	glClearColor(52.9, 80.8, 92.2, 1.0);
	cout << "Initialization successfull" << endl;

	//get the initial time
	lastTime = glfwGetTime();
	// lastTime = (float)glutGet(GLUT_ELAPSED_TIME);
}



//render fullscreen quad using the quad vertex array object
void DrawFullScreenQuad() {
	//bind the quad vertex array object
	glBindVertexArray(quadVAOID);
	//draw two triangles
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
}

//release all allocated resources
void OnShutdown() {

	glDeleteVertexArrays(1, &quadVAOID);
	glDeleteBuffers(1, &quadVBOID);
	glDeleteBuffers(1, &quadIndicesID);

	//delete all textures
	glDeleteTextures(1, &textureID);

	//delete all meshes
	// size_t total_meshes = meshes.size();
	// for (size_t i = 0; i < total_meshes; i++) {
	// 	delete meshes[i];
	// 	meshes[i] = 0;
	// }
	// meshes.clear();

	// size_t total_materials = materials.size();
	// for( size_t i=0;i<total_materials;i++) {
	// 	delete materials[i];
	// 	materials[i] = 0;
	// }
	materials.clear();

	//Destroy shader
	shader.DeleteShaderProgram();
	pathtraceShader.DeleteShaderProgram();
	flatShader.DeleteShaderProgram();

	//Destroy vao and vbo
	glDeleteBuffers(1, &vboVerticesID);
	glDeleteBuffers(1, &vboIndicesID);
	glDeleteVertexArrays(1, &vaoID);

	glDeleteVertexArrays(1, &lightVAOID);
	glDeleteBuffers(1, &lightVerticesVBO);

	glDeleteTextures(1, &texVerticesID);
	glDeleteTextures(1, &texTrianglesID);
	cout << "Shutdown successfull" << endl;
}

//resize event handler
// void OnResize(int w, int h) {
// 	//set the viewport
// 	glViewport (0, 0, (GLsizei) w, (GLsizei) h);
// 	//setup the projection matrix
// 	P = glm::perspective(glm::radians(60.0f),(float)w/h, 0.1f,1000.0f);
// }

int x = 1;
glm::vec3 eye = glm::vec3(15.0,  10.0, -10.0) ;
//display callback function
void renderizar( float **fpI, float **fpHisto) {

	//FPS calculation
	++total_frames;


	// if((current - lastTime) > 1000) {
	// 	fps = 1000.0f*total_frames/(current-lastTime);
	// 	std::cout<<"FPS: "<<fps<<std::endl;
	// 	lastTime= current;
	// 	total_frames = 0;
	// }

	// float current = 1.0;

	//clear colour and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//set the camera transformation
	// P = glm::perspective(glm::radians(60.0f),(float)WIDTH/HEIGHT, 0.1f,1000.0f);
	// glm::mat4 T	= glm::translate(glm::mat4(1.0f),glm::vec3(0.0f, 0.0f, dist));
	// glm::mat4 Rx	= glm::rotate(T,  rX, glm::vec3(1.0f, 0.0f, 0.0f));
	// glm::mat4 MV = glm::rotate(Rx, rY, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
	glm::mat4 P    = glm::perspective(glm::radians(60.0f), (float)WIDTH / HEIGHT, 0.1f, 1000.0f);
	glm::mat4 model;
	// model = glm::scale(model, glm::vec3(0.4, 0.4, 0.4) );

	lightPosOS = glm::vec3(55.0, 35.0, -10.0);
	MV = view * model;


	//get the eye position and inverse of MVP matrix
	// glm::mat4 invMV  = glm::inverse(MV);
	// glm::vec3 eyePos = glm::vec3(invMV[3][0]*0.4, invMV[3][1]*0.4, invMV[3][2]*0.4);
	glm::vec3 eyePos = eye / 0.3f;
	glm::mat4 invMVP = glm::inverse(P * MV);


  
	
	//set the pathtracing shader
	pathtraceShader.Use();
	glUniform1f(pathtraceShader("spp"), 1.0);

	//pass shader uniforms
	glUniform3fv(pathtraceShader("eyePos"), 1, glm::value_ptr(eyePos));

	glUniform3fv(pathtraceShader("light_position"), 1, &(lightPosOS.x));
	glUniformMatrix4fv(pathtraceShader("invMVP"), 1, GL_FALSE, glm::value_ptr(invMVP));



	cout<<"OK"<<endl;
	// Image Writing
	float* imageData = (float*) malloc(WIDTH * HEIGHT * 3*sizeof(float));
	unsigned char* sampleImageData = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

	for (int i = 0; i < WIDTH * HEIGHT * 3 ; ++i)
		imageData[i] = 0.0;

	namedWindow( "hello", CV_WINDOW_AUTOSIZE );

    int nc_h = 61;
	fpHisto = new float*[nc_h];
    for (int i=0; i < nc_h; i++){
        fpHisto[i] = new float[WIDTH * HEIGHT]; // &fpH[i * nx_h*ny_h];        
    }

    // initialize histogram
    for (int i=0; i < nc_h; i++)
    	for (int j=0; j < WIDTH * HEIGHT; j++)
    		fpHisto[i][j] = 0.0;




	float spp = 20.0;
	for (int s = 0; s < spp; ++s)
	{  
		//draw a fullscreen quad
		glDisable(GL_DEPTH_TEST);
		glClear(GL_COLOR_BUFFER_BIT);

		// float current = glfwGetTime();
		// cout << "current: " << current << endl;
		glUniform1i(pathtraceShader("s"), s);
		glUniform1f (pathtraceShader("time"), float(s)/100 + 0.01 );
		DrawFullScreenQuad();
		glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, sampleImageData);

		for (unsigned int y = 0; y < HEIGHT; ++y)
			for (unsigned int x = 0; x < WIDTH; ++x) {

				int offset = y * WIDTH  + x ;
				int offset2 = (HEIGHT - 1 - y) * WIDTH * 3 + x * 3;

				//RGB
				for (int c = 0; c < 3; ++c){
					unsigned char channelValue = sampleImageData[offset2 + c];

					for (int bin = 0; bin < 20; ++bin) //bins
					{
						if(channelValue >= 12.75*bin && channelValue <= 12.75*(bin+1)) {
							fpHisto[bin + 20*c][offset] += 1.0;
							break;
						}
					}
				}
				fpHisto[60][offset]=spp; //total sample contribution
			}		
	}

	glUniform1f(pathtraceShader("spp"), spp);

	//draw a fullscreen quad
		glDisable(GL_DEPTH_TEST);
		glClear(GL_COLOR_BUFFER_BIT);

		// float current = glfwGetTime();
		// cout << "current: " << current << endl;
		// glUniform1i(pathtraceShader("s"), s);
		// glUniform1f (pathtraceShader("time"), float(s)/100 + 0.01 );
		DrawFullScreenQuad();
		glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, sampleImageData);


		//*********** IMAGE 
    fpI = new float*[3];
    
    for (int ii=0; ii < 3; ii++) {       
        fpI[ii] = new float[WIDTH*HEIGHT];        
    }

		for (unsigned int y = 0; y < HEIGHT; ++y)
			for (unsigned int x = 0; x < WIDTH; ++x) {

				int offset = y * WIDTH  + x ;
				int offset2 = (HEIGHT - 1 - y) * WIDTH * 3 + x * 3;


				fpI[0][offset] = float(sampleImageData[offset2 + 0]);///255.0;
				fpI[1][offset] = float(sampleImageData[offset2 + 1]);///255.0;
				fpI[2][offset] = float(sampleImageData[offset2 + 2]);///255.0;

				//RGB to BRG
				// data[offset + 0] = sampleImageData[offset2 + 2];
				// data[offset + 1] = sampleImageData[offset2 + 1];
				// data[offset + 2] = sampleImageData[offset2 + 0];
			}



		cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    unsigned char* data = img.ptr();

    for (unsigned int y = 0; y < HEIGHT; ++y)
			for (unsigned int x = 0; x < WIDTH; ++x) {

				int offset = y * WIDTH  + x ;
				int offset2 = y * WIDTH *3 + x *3;

				//RGB to BRG
				data[offset2 + 0] = fpI[2][offset];
				data[offset2 + 1] = fpI[1][offset];
				data[offset2 + 2] = fpI[0][offset];
			}

    // namedWindow( "hello", CV_WINDOW_AUTOSIZE );
    imwrite("pathTraceImage.png",img) ;
    // waitKey(0);








		//*** Histograma
	// printf("\n\n");
 //    float sum = 0.0;
 //    for (int ii=0; ii < 61; ii++){
 //        printf("%f  ", fpHisto[ii][WIDTH*400 + 50 ]);
 //        if (ii >=0 && ii <20 )
 //        {
 //            sum += fpHisto[ii][WIDTH*400  + 50];
 //        }
 //        if (ii ==19)
 //            printf("\n");
 //    }
 //    printf("sum: %f\n", sum);
 //    printf("\n");
	







	//unbind pathtracing shader
	pathtraceShader.UnUse();


	//swap front and back buffers to show the rendered result
	// glfwSwapBuffers(window);
	// glfwPollEvents();
}

void generateInputs(float **fpI, float **fpHisto) {

	// Init GLFW
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "GPU pathtracer - OpenGL 3.3", nullptr, nullptr); // Windowed
	glfwMakeContextCurrent(window);

	// Set the required callback functions
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// Options
	// glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Initialize GLEW to setup the OpenGL Function pointers
	glewExperimental = GL_TRUE;
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, WIDTH, HEIGHT);



	//initialize glew
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)	{
		cerr << "Error: " << glewGetErrorString(err) << endl;
	} else {
		if (GLEW_VERSION_3_3)
		{
			cout << "Driver supports OpenGL 3.3\nDetails:" << endl;
		}
	}
	err = glGetError(); //this is to ignore INVALID ENUM error 1282
	GL_CHECK_ERRORS


	//output hardware information
	cout << "\tUsing GLEW " << glewGetString(GLEW_VERSION) << endl;
	cout << "\tVendor: " << glGetString (GL_VENDOR) << endl;
	cout << "\tRenderer: " << glGetString (GL_RENDERER) << endl;
	cout << "\tVersion: " << glGetString (GL_VERSION) << endl;
	cout << "\tGLSL: " << glGetString (GL_SHADING_LANGUAGE_VERSION) << endl;

	GL_CHECK_ERRORS


	// int texture_width, texture_height,channels;
	// GLubyte* pData = SOIL_load_image("B.png", &texture_width, &texture_height, &channels, SOIL_LOAD_AUTO);

	//OpenGL initialization
	initialize();
	renderizar(fpI, fpHisto);

}


// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	if (action == GLFW_PRESS)
		keys[key] = true;
	else if (action == GLFW_RELEASE)
		keys[key] = false;

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		eye[0] -= 0.4;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		eye[1] -= 0.4;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		eye[2] -= 0.4;


}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
}
