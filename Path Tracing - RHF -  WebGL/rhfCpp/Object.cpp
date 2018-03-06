// #include "Object.h"  
 
 
// ObjLoader::ObjLoader() {

// }

// ObjLoader::~ObjLoader() {

// }

// #include <fstream>
// #include <sstream>

// std::string trim(const std::string& str,
//                  const std::string& whitespace = " \t")
// {
//     const auto strBegin = str.find_first_not_of(whitespace);
//     if (strBegin == std::string::npos)
//         return ""; // no content

//     const auto strEnd   = str.find_last_not_of(whitespace);
//     const auto strRange = strEnd - strBegin + 1;

//     return str.substr(strBegin, strRange);
// }

// bool ReadMaterialLibrary(const std::string& filename, vector<Material*>& materials) {
// 	ifstream fp(filename.c_str(),ios::in);
// 	if(!fp)
// 		return false;
//     string tmp(std::istreambuf_iterator<char>(fp), (std::istreambuf_iterator<char>()));
// 	istringstream buffer(tmp);
// 	fp.close();
	  
// 	//now parse the file
// 	string line;
// 	Material* pMat = 0;
// 	printf("%s\n", "ReadMaterialLibrary");
// 	while(getline(buffer, line)) { 
// 		line = trim(line);

// 		if(line.find_first_of("#") != string::npos) //its a comment leave it
// 			continue;
// 		if(line.length()==0)
// 			continue;
		
// 		int space_index = line.find_first_of(" ");
//         string prefix = trim(line.substr(0, space_index)); 

// 		if(prefix.compare("newmtl") ==0) //if we have a newmtl block
// 		{
// 			pMat = new Material(); 
// 			pMat->name = line.substr(space_index+1);
// 			materials.push_back(pMat);
// 		}



// 		else if(prefix.compare("map_Kd") == 0) {
// 			pMat->map_Kd = line.substr(space_index+1); 
// 		}
// 	}

// 	return true;
// }

// bool ObjLoader::Load(const string& filename, 
// 	            	 vector<Material*>& materials, 
// 					 BBox& bbox, vector<glm::vec3>& verts2, 
// 					 vector<unsigned short>& indices2) {

// 	ifstream fp(filename.c_str(), ios::in);
// 	if(!fp)
// 		return false;
//     string tmp(std::istreambuf_iterator<char>(fp), (std::istreambuf_iterator<char>()));
// 	istringstream buffer(tmp);
// 	fp.close();

// 	float min[3]={1000,1000,1000}, max[3]={-1000, -1000, -1000};

// 	string line;
// 	Mesh* mesh = 0;
// 	Material* pMat = 0;
	
// 	bool hasNormals = false;
// 	bool hasUVs     = false;
// 	bool isNewMesh  = true; 
// 	vector<glm::vec3> normals;
// 	vector<glm::vec2> uvs; 

// 	while(getline(buffer, line)) {
// 		line=trim(line);
// 		if(line.find_first_of("#") != -1) //its a comment leave it
// 			continue;

// 		int space_index = line.find_first_of(" ");
//         string prefix = trim(line.substr(0, space_index));
// 		if(prefix.length()==0)
// 			continue;

// 		if(prefix.compare("vt")==0) //if we have a texture coord
// 		{
// 			line = line.substr(space_index+1);
// 			glm::vec2 uv;

// 			istringstream s(line);
// 			s>>uv.x;
// 			s>>uv.y;

// 			uvs.push_back(uv);
// 			hasUVs = true;
// 		}

// 		if(prefix.compare("v")==0) //if we have a vertex
// 		{
// 			if (isNewMesh) {
// 				mesh = new Mesh();
// 				isNewMesh = false;
// 			}

// 			line = line.substr(space_index+1);
// 			glm::vec3 v;

// 			istringstream s(line);
// 			s>>v.x;
// 			s>>v.y;
// 			s>>v.z;
// 			if(v.x<min[0])
// 				min[0] = v.x;
// 			if(v.y<min[1])
// 				min[1] = v.y;
// 			if(v.z<min[2])
// 				min[2] = v.z;

// 			if(v.x>max[0])
// 				max[0] = v.x;
// 			if(v.y>max[1])
// 				max[1] = v.y;
// 			if(v.z>max[2])
// 				max[2] = v.z;
// 			verts2.push_back(v);
// 		}

// 		if(prefix.compare("vn")==0) //if we have a vertex
// 		{ 
// 			line = line.substr(space_index+1);
// 			glm::vec3 v;

// 			istringstream s(line);
// 			s>>v.x;
// 			s>>v.y;
// 			s>>v.z; 
// 			normals.push_back(v);
// 			hasNormals = true;
// 		}

// 		if(prefix.compare("f")==0)
// 		{
// 			line = line.substr(space_index+1);
// 			Face f;
// 			int index = space_index;
// 			int start = 0;
// 			string face_data;
// 			string normal_data;
// 			string uv_data;
// 			string l2="";

// 			int space_index = line.find_first_of(" ", start+1);
// 			int count = 1;

// 			while(space_index!= -1) {
// 				l2 = line.substr(start, space_index-start);
// 				int firstSlashIndex = l2.find("/");
// 				int secondSlashIndex = l2.find("/",firstSlashIndex+1);

// 				face_data.append(l2.substr(0, firstSlashIndex) );
// 				face_data.append(" ");
// 				if(hasUVs) {
// 					uv_data.append(l2.substr(firstSlashIndex+1,(secondSlashIndex-firstSlashIndex)-1) );
// 					uv_data.append(" ");
// 				}
// 				if(hasNormals) {
// 					normal_data.append(l2.substr(secondSlashIndex+1) );
// 					normal_data.append(" ");
// 				}
// 				start  = space_index;
// 				space_index = line.find_first_of(" ", start+1);
// 				++count;
// 			}

// 			l2 = line.substr(line.find_last_of(" "));
// 			int firstSlashIndex  = l2.find("/");
// 			int secondSlashIndex = l2.find("/",firstSlashIndex+1);

// 			face_data.append(l2.substr(0, firstSlashIndex));
// 			if(hasUVs) {
// 				uv_data.append(l2.substr(firstSlashIndex+1,(secondSlashIndex-firstSlashIndex)-1) );
// 			}
// 			if(hasNormals) {
// 				normal_data.append(l2.substr(secondSlashIndex+1) );
// 			}

// 			istringstream s(face_data);

// 			s>>f.a;
// 			s>>f.b;
// 			s>>f.c;
// 			f.a-=1; //indices start from 0
// 			f.b-=1;
// 			f.c-=1;
			 
// 			indices2.push_back(f.a);
// 			indices2.push_back(f.b);
// 			indices2.push_back(f.c);

// 			if (materials[mesh->material_index]->map_Kd != "")
// 				indices2.push_back(mesh->material_index);
// 			else {
// 				printf("%s\n", "ok");
// 				indices2.push_back(255);
// 			}

// 			istringstream n(normal_data);
			
// 			n>>f.d; //indices start from 0
// 			n>>f.e;
// 			n>>f.f;
// 			f.d-=1;
// 			f.e-=1;
// 			f.f-=1;

			 		
// 			istringstream uv(uv_data);

// 			uv>>f.g; //indices start from 0
// 			uv>>f.h;
// 			uv>>f.i;
// 			f.g-=1;
// 			f.h-=1;
// 			f.i-=1;
			 
			 

		 
// 			if(mesh->material_index != -1) {
// 				materials[mesh->material_index]->sub_indices.push_back(f.a);
// 				materials[mesh->material_index]->sub_indices.push_back(f.b);
// 				materials[mesh->material_index]->sub_indices.push_back(f.c);
// 				materials[mesh->material_index]->sub_indices.push_back(f.d);
// 				materials[mesh->material_index]->sub_indices.push_back(f.e);
// 				materials[mesh->material_index]->sub_indices.push_back(f.f);
// 				materials[mesh->material_index]->sub_indices.push_back(f.g);
// 				materials[mesh->material_index]->sub_indices.push_back(f.h);
// 				materials[mesh->material_index]->sub_indices.push_back(f.i);
// 			} 
// 		}

// 		if(prefix.compare("mtllib")==0) {
// 			//we have a material library
// 			std::string full_path = filename.substr(0, filename.find_last_of("/")+1);
// 			line = line.substr(line.find_first_of("mtllib")+7);
// 			full_path.append(line);
// 			ReadMaterialLibrary(full_path, materials);
// 		}

// 		if(prefix.compare("usemtl")==0) {
// 			string material_name = line.substr(space_index+1); 
// 			int index = -1;

// 			//BUSCAR EL INDICE DEL MATERIAL
// 			for(size_t i = 0; i<materials.size(); i++) {
// 				if(materials[i]->name.compare(material_name) ==0) {
// 					index = i;
// 					break;
// 				}
// 			}

// 			mesh->material_index = index;
// 		}

// 		if(prefix.compare("o")==0) {
// 			// mesh->name = line.substr(space_index+1); 	
// 			// printf("%s\n", mesh->name.c_str());
// 			isNewMesh = true;
// 		}
// 	}
// 	printf("%s\n", "ALL GOOD?");
	   
// 	bbox.min.x = min[0];
// 	bbox.min.y = min[1];
// 	bbox.min.z = min[2];

// 	bbox.max.x = max[0];
// 	bbox.max.y = max[1];
// 	bbox.max.z = max[2];

// 	return true;
// }




#include "Object.h"  
 
 
ObjLoader::ObjLoader() {

}

ObjLoader::~ObjLoader() {

}

#include <fstream>
#include <sstream>

std::string trim(const std::string& str,
                 const std::string& whitespace = " \t")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

bool ReadMaterialLibrary(const std::string& filename, vector<Material*>& materials) {
	ifstream fp(filename.c_str(),ios::in);
	if(!fp)
		return false;
    string tmp(std::istreambuf_iterator<char>(fp), (std::istreambuf_iterator<char>()));
	istringstream buffer(tmp);
	fp.close();
	  
	//now parse the file
	string line;
	Material* pMat = 0;
	while(getline(buffer, line)) { 
		line = trim(line);

		if(line.find_first_of("#") != string::npos) //its a comment leave it
			continue;
		if(line.length()==0)
			continue;
		
		int space_index = line.find_first_of(" ");
        string prefix = trim(line.substr(0, space_index)); 

		if(prefix.compare("newmtl") ==0) //if we have a newmtl block
		{
			pMat = new Material(); 
			pMat->name = line.substr(space_index+1);
			materials.push_back(pMat);
		}
		  
		// else if(prefix.compare("Ns")==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Ns;
		// } 
		// else if(prefix.compare("Ni")==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Ni;
		// }

		// else if(prefix.compare("d")== 0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->d;
		// }

		// else if(prefix.compare("Tr") ==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Tr;
		// }

		// else if(prefix.compare("Tf")==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Tf[0]>>pMat->Tf[1]>>pMat->Tf[2];
		// }

		// else if(prefix.compare("illum")==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->illum;
		// }

		// else if(prefix.compare("Ka")==0) { //0.5880 0.5880 0.5880
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Ka[0]>>pMat->Ka[1]>>pMat->Ka[2];
		// }

		// else if(prefix.compare("Kd")==0) {// 0.5880 0.5880 0.5880
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Kd[0]>>pMat->Kd[1]>>pMat->Kd[2];
		// }

		// else if(prefix.compare("Ks")==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Ks[0]>>pMat->Ks[1]>>pMat->Ks[2];
		// }

		// else if(prefix.compare("Ke")==0) {
		// 	line = line.substr(space_index+1);
		// 	istringstream s(line);
		// 	s>>pMat->Ke[0]>>pMat->Ke[1]>>pMat->Ke[2];
		// }
		// else if(prefix.compare("map_Ka")==0) {
		// 	pMat->map_Ka = line.substr(space_index+1); 
		// }
		else if(prefix.compare("map_Kd") == 0) {
			pMat->map_Kd = line.substr(space_index+1); 
		}
	}

	return true;
}

bool ObjLoader::Load(const string& filename, vector<Material*>& materials, 
	BBox& bbox, vector<glm::vec3>& verts2, vector<unsigned short>& indices2) {
	ifstream fp(filename.c_str(),ios::in);
	if(!fp)
		return false;
    string tmp(std::istreambuf_iterator<char>(fp), (std::istreambuf_iterator<char>()));
	istringstream buffer(tmp);
	fp.close();

	float min[3]={1000,1000,1000}, max[3]={-1000, -1000, -1000};

	string line;
	Mesh* mesh = 0;
	Material* pMat = 0;
	
	bool hasNormals = false;
	bool hasUVs = false;
	bool isNewMesh = true; 
	vector<glm::vec3> normals;
	vector<glm::vec2> uvs; 
	int total_triangles = 0; 

	while(getline(buffer, line)) {
		line=trim(line);
		if(line.find_first_of("#") != -1) //its a comment leave it
			continue;

		int space_index = line.find_first_of(" ");
        string prefix = trim(line.substr(0, space_index));
		if(prefix.length()==0)
			continue;

		if(prefix.compare("vt")==0) //if we have a texture coord
		{
			line = line.substr(space_index+1);
			glm::vec2 uv;

			istringstream s(line);
			s>>uv.x;
			s>>uv.y;

			uvs.push_back(uv);
			hasUVs = true;
		}

		if(prefix.compare("v")==0) //if we have a vertex
		{
			if(isNewMesh) {
				mesh = new Mesh();
				// meshes.push_back(mesh);
				isNewMesh = false;
			}

			line = line.substr(space_index+1);
			glm::vec3 v;

			istringstream s(line);
			s>>v.x;
			s>>v.y;
			s>>v.z;
			if(v.x<min[0])
				min[0] = v.x;
			if(v.y<min[1])
				min[1] = v.y;
			if(v.z<min[2])
				min[2] = v.z;

			if(v.x>max[0])
				max[0] = v.x;
			if(v.y>max[1])
				max[1] = v.y;
			if(v.z>max[2])
				max[2] = v.z;
			verts2.push_back(v);
		}
		if(prefix.compare("vn")==0) //if we have a vertex
		{ 
			line = line.substr(space_index+1);
			glm::vec3 v;

			istringstream s(line);
			s>>v.x;
			s>>v.y;
			s>>v.z; 
			normals.push_back(v);
			hasNormals = true;
		}
		if(prefix.compare("f")==0)
		{
			line = line.substr(space_index+1);
			Face f;
			int index  = space_index;
			int start=0;
			string face_data;
			string normal_data;
			string uv_data;
			string l2="";
			int space_index = line.find_first_of(" ", start+1);
			int count = 1;
			while(space_index!= -1) {
				l2 = line.substr(start, space_index-start);
				int firstSlashIndex = l2.find("/");
				int secondSlashIndex = l2.find("/",firstSlashIndex+1);

				face_data.append(l2.substr(0, firstSlashIndex) );
				face_data.append(" ");
				if(hasUVs) {
					uv_data.append(l2.substr(firstSlashIndex+1,(secondSlashIndex-firstSlashIndex)-1) );
					uv_data.append(" ");
				}
				if(hasNormals) {
					normal_data.append(l2.substr(secondSlashIndex+1) );
					normal_data.append(" ");
				}
				start  = space_index;
				space_index = line.find_first_of(" ", start+1);
				++count;
			}
			l2 = line.substr(line.find_last_of(" "));
			int firstSlashIndex = l2.find("/");
			int secondSlashIndex = l2.find("/",firstSlashIndex+1);

			face_data.append(l2.substr(0, firstSlashIndex));
			if(hasUVs) {
				uv_data.append(l2.substr(firstSlashIndex+1,(secondSlashIndex-firstSlashIndex)-1) );
			}
			if(hasNormals) {
				normal_data.append(l2.substr(secondSlashIndex+1) );
			}
			istringstream s(face_data);

			s>>f.a;
			s>>f.b;
			s>>f.c;
			f.a-=1;
			f.b-=1;
			f.c-=1;
			 
			indices2.push_back(f.a);
			indices2.push_back(f.b);
			indices2.push_back(f.c);

            //si es un index valido
			if (materials[mesh->material_index]->map_Kd != "")
				indices2.push_back(mesh->material_index);
			else
				indices2.push_back(255);

			istringstream n(normal_data);
			
			n>>f.d;
			n>>f.e;
			n>>f.f;
			f.d-=1;
			f.e-=1;
			f.f-=1;

			 		
			istringstream uv(uv_data);
			uv>>f.g;
			uv>>f.h;
			uv>>f.i;
			f.g-=1;
			f.h-=1;
			f.i-=1;
			 
			total_triangles++;
			 

		 
			// if(mesh->material_index != -1) {
			// 	materials[mesh->material_index]->sub_indices.push_back(f.a);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.b);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.c);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.d);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.e);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.f);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.g);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.h);
			// 	materials[mesh->material_index]->sub_indices.push_back(f.i);
			// } 

			if(count==4) {
				unsigned short tmpP = 0;
				unsigned short tmpT = 0;
				unsigned short tmpN = 0;
				s>>tmpP;
				uv>>tmpT;
				n>>tmpN;

				tmpP=tmpP-1;
				f.b = f.c;
				f.c = tmpP;

				indices2.push_back(f.a);
				indices2.push_back(f.b);
				indices2.push_back(f.c);
				
				if(materials[mesh->material_index]->map_Kd!="")
					indices2.push_back(mesh->material_index);
				else
					indices2.push_back(255);

				tmpN=tmpN-1;
				f.e=f.f;
				f.f=tmpN;

				
				tmpT=tmpT-1;
				f.h=f.i;
				f.i=tmpT;
				 
				total_triangles++;
				// if(mesh->material_index != -1) {
				// 	materials[mesh->material_index]->sub_indices.push_back(f.a);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.b);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.c);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.d);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.e);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.f);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.g);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.h);
				// 	materials[mesh->material_index]->sub_indices.push_back(f.i);
				// } 
			} 
		}

		if(prefix.compare("mtllib")==0) { //once
			//we have a material library
			std::string full_path = filename.substr(0, filename.find_last_of("/")+1);
			line = line.substr(line.find_first_of("mtllib")+7);
			full_path.append(line);
			ReadMaterialLibrary(full_path, materials);
		}

		if(prefix.compare("usemtl")==0) {  //many
			string material_name = line.substr(space_index+1); 
			int index = -1;

			//Looking for the index
			for(size_t i=0; i < materials.size();i++) {  
				if(materials[i]->name.compare(material_name) ==0) {
					index = i;
					break;
				}
			}
			mesh->material_index = index;
		}

		if(prefix.compare("g")==0) {
			mesh->name = line.substr(space_index+1); 	
			isNewMesh = true;
		}
	}
	   
	bbox.min.x = min[0];
	bbox.min.y = min[1];
	bbox.min.z = min[2];

	bbox.max.x = max[0];
	bbox.max.y = max[1];
	bbox.max.z = max[2];

	// verts.resize( total_triangles * 3);

	// int count=0;
	// int count2=0;
	// int sub_count =0;

	// //sort meshes by material 
	// for(size_t i=0;i<materials.size();i++) {
	// 	Material* pMat = materials[i];
	// 	pMat->offset = count;
	// 	for(size_t j=0;j<pMat->sub_indices.size();j+=9) {
	// 		verts[count].pos	 =   verts2[  pMat->sub_indices[j ] ];
	// 		verts[count].normal	 =	normals[  pMat->sub_indices[j+3]  ];
	// 		verts[count++].uv	 =		uvs[  pMat->sub_indices[j+6]  ];				 

	// 		verts[count].pos	 =   verts2[  pMat->sub_indices[j   + 1] ];
	// 		verts[count].normal	 =	normals[  pMat->sub_indices[j+3 + 1]  ];
	// 		verts[count++].uv	 =		uvs[  pMat->sub_indices[j+6 + 1]  ];				 
			 
	// 		verts[count].pos	 =   verts2[  pMat->sub_indices[j	+ 2]  ];
	// 		verts[count].normal	 =	normals[  pMat->sub_indices[j+3 + 2]  ];
	// 		verts[count++].uv	 =		uvs[  pMat->sub_indices[j+6 + 2]  ];		 
						
	// 		indices.push_back(count2++);
	// 		indices.push_back(count2++);
	// 		indices.push_back(count2++);
	// 		sub_count+=3;
	// 	} 
	// 	pMat->count = sub_count;
	// 	sub_count = 0;

	// }
	 
	 
	//copy the contents into a linear array without material sorting
	/*
	for(size_t i=0;i<inds.size();i+=9) { 
		
		verts[count].pos	= vertices[inds[i]];
		verts[count].normal = normals[inds[i+3]];
		verts[count++].uv	= uvs[inds[i+6]];				 

		verts[count].pos	= vertices[inds[i+1]];
		verts[count].normal = normals[inds[(i+3)+1]];
		verts[count++].uv	= uvs[inds[(i+6)+1]];				 

		verts[count].pos	= vertices[inds[i+2]];
		verts[count].normal = normals[inds[(i+3)+2]];
		verts[count++].uv	= uvs[inds[(i+6)+2]];	 

		indices.push_back(count2++);
		indices.push_back(count2++);
		indices.push_back(count2++);
	}	 
	 */

	return true;
}


