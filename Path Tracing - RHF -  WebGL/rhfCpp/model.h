
#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <stdio.h>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


#include <glm/glm.hpp>
using namespace glm;
using namespace std;

class BoundingBox
{
public:
	glm::vec3 min, max;
	BoundingBox() {
		min = glm::vec3(1000, 1000, 1000);
		max = glm::vec3(-1000, -1000, -1000);
	}

};


class Material_ {
public:
	float ambient[3];
	float diffuse[3];
	float specular[3];
	float Tf[3];
	int   illum;
	float Ka[3];
	float Kd[3];
	float Ks[3];
	float Ke[3];
	std::string map_Ka,  map_Kd, name; 
	float Ns, Ni, d, Tr; 
	vector<unsigned short> sub_indices;
	int offset;
	int count;

};


class Model {
	// The permutation vector
public:
	std::vector< unsigned short > vertexIndices, normalIndices;
	std::vector< vec3 > temp_vertices;
	std::vector< vec3 > temp_normals;



	BoundingBox box;
	std::vector < vec3 > vertices;
	std::vector < vec3 > normals;

	vector<Material_> materials;

	int meshMaterialIndex;
	string meshName;
	// std::vector < vec3 > indices;

	// Initialize with the reference values for the permutation vector
	Model(std::string path) ;
	bool ReadMaterialLibrary(char* filename, vector<Material_>& materials);
	std::string trim(const std::string& str, const std::string& whitespace = " \t");
};


#endif