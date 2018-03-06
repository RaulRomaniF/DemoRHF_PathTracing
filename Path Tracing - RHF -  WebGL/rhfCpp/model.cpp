#include "model.h"


// std::string Model::trim(const std::string& str,
//                  const std::string& whitespace = " \t")
// {
//     const auto strBegin = str.find_first_not_of(whitespace);
//     if (strBegin == std::string::npos)
//         return ""; // no content

//     const auto strEnd = str.find_last_not_of(whitespace);
//     const auto strRange = strEnd - strBegin + 1;

//     return str.substr(strBegin, strRange);
// }


bool Model::ReadMaterialLibrary(char* filename, vector<Material_>& materials) {

  string path = "model/"+ std::string(filename);

	FILE* file = fopen( path.c_str(), "r");

	if ( file == NULL ) {
		printf("Impossible to open the file !\n");
	} else {
		printf("MATERIAL LOADED :)\n");
	}

    Material_ material_temp;
    // material_temp.name = "07___Default";
    // materials.push_back(material_temp);
	while ( 1 ) {

		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.



		if ( strcmp( lineHeader, "newmtl" ) == 0 ) {

			char materialName[128];

			fscanf(file, "%s", materialName);

			material_temp.name = string(materialName);

		}
		else if ( strcmp( lineHeader, "map_Kd" ) == 0 ) {
			vec3 normal;

			char map_kd[128];
			fscanf(file, "%s\n", map_kd );
			temp_normals.push_back(normal);

			material_temp.map_Kd = string(map_kd);

			materials.push_back(material_temp);
		}
	}

	return true;
}


Model::Model(std::string path) {
	FILE * file = fopen(path.c_str(), "r");


	if ( file == NULL ) {
		printf("Impossible to open the file !\n");
	} else {
		printf("OBJ LOADED\n");
	}


    int count = 0;
	while ( 1 ) {

		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		if ( strcmp( lineHeader, "v" ) == 0 ) {
			vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );


			if (vertex.x < box.min.x)
				box.min.x = vertex.x;

			if (vertex.y < box.min.y)
				box.min.y = vertex.y;

			if (vertex.z < box.min.z)
				box.min.z = vertex.z;

			if (vertex.x > box.max.x)
				box.max.x = vertex.x;
			if (vertex.y > box.max.y)
				box.max.y = vertex.y;
			if (vertex.z > box.max.z)
				box.max.z = vertex.z;

			temp_vertices.push_back(vertex);
		}
		else if ( strcmp( lineHeader, "vn" ) == 0 ) {
			vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
			temp_normals.push_back(normal);
		} else if ( strcmp( lineHeader, "f" ) == 0 ) {
			std::string vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uv[3], normalIndex[3];

			// int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uv[0], &normalIndex[0], &vertexIndex[1], &uv[1], &normalIndex[1], &vertexIndex[2],  &uv[2], &normalIndex[2] );
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uv[0], &normalIndex[0], &vertexIndex[1], &uv[1], &normalIndex[1], &vertexIndex[2],  &uv[2], &normalIndex[2] );
			if (matches != 9) {
				cout << "matches: " << matches << endl;
				printf("File can't be read by our simple parser : ( Try exporting with other options\n");
			}

			vertexIndices.push_back(vertexIndex[0] - 1);
			vertexIndices.push_back(vertexIndex[1] - 1);
			vertexIndices.push_back(vertexIndex[2] - 1);

			if(meshMaterialIndex != -1 && materials[meshMaterialIndex].map_Kd != "" ){
				vertexIndices.push_back(meshMaterialIndex); //index material
				// vertexIndices.push_back(5); //index material
			}else
			    vertexIndices.push_back(255); //index material

			// normalIndices.push_back(normalIndex[0] - 1);
			// normalIndices.push_back(normalIndex[1] - 1);
			// normalIndices.push_back(normalIndex[2] - 1);
		}

		else if(strcmp( lineHeader, "mtllib" ) == 0 ) { //once
			//we have a material library

            char full_path[128] ;
			fscanf(file, "%s \n", full_path);
			ReadMaterialLibrary(full_path, materials);

		}

		else if(strcmp( lineHeader, "usemtl" ) == 0 ) { //many

			char materialName_cstr[128] ;
			fscanf(file, "%s \n", materialName_cstr);

			int index = -1;
			//Looking for the index
			for(size_t i=0; i < materials.size(); i++) {  
				if(materials[i].name.compare(string(materialName_cstr)) == 0) {
					index = i;
					break;
				}
			}
			meshMaterialIndex = index;

			cout<<"index: "<<index<<endl;
			// mesh->material_index = index;
		}

		else if(strcmp( lineHeader, "o" ) == 0 ) { //once

			char meshName_cstr[128] ;

			fscanf(file, "%s \n", meshName_cstr);

			meshName = string(meshName_cstr);

			cout<<"G: "<<meshName<<endl;


			// meshName = line.substr(space_index+1); 	
			// isNewMesh = true;
		}
	}

	cout << "All good." << endl;



	// Para cada vértice de cada triángulo
	// for ( unsigned int i = 0; i < vertexIndices.size(); i++ ) {
	// 	vec3 vertex = temp_vertices[ vertexIndices[i] ];
	// 	vec3 normal = temp_normals [ normalIndices[i] ];
	// 	vertices.push_back(vertex);
	// 	normals.push_back(normal);
	// }










	//    printf("%d \n", vertices.size());
	// printf("%d \n", normals.size());

	// for (int i = 0; i < temp_vertices.size(); ++i)
	// {
	// 	printf("%f %f %f\n", temp_vertices[i].x, temp_vertices[i].y, temp_vertices[i].z );
	// }
}
