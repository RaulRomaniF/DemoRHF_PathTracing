#version 300 es 

precision highp float;
precision highp sampler2DArray;
precision highp isampler2D;

out vec4 vFragColor; //fragment shader output


//structs for Ray, Box and Camera objects
struct Ray { 
	vec3 origin, dir;
} eyeRay; 

struct Box { 
	vec3 min, max; 
};
struct Camera {
   vec3 U,V,W; 
   float d;
}cam;


//input from the vertex shader
// smooth in vec2 vUV;					//interpolated texture coordinates
in vec2 vUV;					//interpolated texture coordinates

//shader uniforms
uniform mat4 invMVP;					//inverse of combined modelview projection matrix
uniform vec4 backgroundColor;			//background colour
uniform vec3 eyePos; 					//eye position in object space

uniform sampler2D vertex_positions;		//mesh vertices
uniform isampler2D triangles_list;		//mesh triangles
uniform sampler2DArray textureMaps;		//all mesh textures




uniform vec3 light_position;			//light position is in object space
uniform Box aabb;	 					//scene's bounding box
uniform float VERTEX_TEXTURE_SIZE; 		//size of the vertex texture
uniform float TRIANGLE_TEXTURE_SIZE; 	//size of the triangle texture 
uniform float time;						//current time

uniform float spp;						//current time

//shader constants
const int MAX_BOUNCES = 2;	//the total number of bounces for each ray

//function to return the intersection of a ray with a box
// returns a vec2 in which the x value contains the t value at the near intersection
// 						the y value contains the t value at the far intersection
vec2 intersectCube(vec3 origin, vec3 ray, Box cube) {		
	vec3   tMin = (cube.min - origin) / ray;		
	vec3   tMax = (cube.max - origin) / ray;		
	vec3     t1 = min(tMin, tMax);		
	vec3     t2 = max(tMin, tMax);

	//max component of the min distance
	float tNear = max(max(t1.x, t1.y), t1.z);

	//min component of the max distance
	float  tFar = min(min(t2.x, t2.y), t2.z);

	return vec2(tNear, tFar);	
}

// //gets the direction given a 2D position and a Camera
vec3 get_direction(vec2 p, Camera c) {
   return normalize(p.x*c.U + p.y*c.V + c.d * c.W);   
}

// //Generates the eye ray for the given camera and a 2D position
void setup_camera(vec2 p) {
 
  eyeRay.origin = eyePos; 
    
  cam.U = (invMVP*vec4(1,0,0,0)).xyz; 
  cam.V = (invMVP*vec4(0,1,0,0)).xyz; 
  cam.W = (invMVP*vec4(0,0,1,0)).xyz; 
  cam.d = 1.0f;    
  
  eyeRay.dir = get_direction(p, cam); 
  eyeRay.dir += p.x*cam.U;
  eyeRay.dir += p.y*cam.V;  
}

// //ray triangle intesection routine. The normal is returned in the given 
// //normal reference argument.
// //
// //The return value is a vec4 with
// //x -> t value at intersection.
// //y -> u texture coordinate
// //z -> v texture coordinate
// //w -> texture map id 
vec4 intersectTriangle(vec3 origin, vec3 dir, int index, out vec3 normal ) {
	                                // texture of width = 38 and heigth = 1
	//                                                <0, 38>/38 = <0, 1> 
	ivec4 list_pos = texture(triangles_list, vec2(  float(float(index) + 0.5f)/TRIANGLE_TEXTURE_SIZE , 0.0f) );

	// float x =  float(float(index) + 0.5f)/TRIANGLE_TEXTURE_SIZE;
	// ivec4 list_pos = texture(triangles_list, vec2(0.0f, 0.0f) );
	
	if((index+1) % 2 != 0 ) {  //if index is impar
		list_pos.xyz = list_pos.zxy;
	}
	
	// // 1D Texture : List of positions        <0, 28>/28 = <0, 1>
	vec3 v0 = texture(vertex_positions, vec2((float(list_pos.z) + 0.5 )/VERTEX_TEXTURE_SIZE, 0.0)).xyz;
	vec3 v1 = texture(vertex_positions, vec2((float(list_pos.y) + 0.5 )/VERTEX_TEXTURE_SIZE, 0.0)).xyz;
	vec3 v2 = texture(vertex_positions, vec2((float(list_pos.x) + 0.5 )/VERTEX_TEXTURE_SIZE, 0.0)).xyz;
	  
	vec3 e1 = v1 - v0; //edges
	vec3 e2 = v2 - v0;

	vec3 tvec = origin - v0;  
	
	vec3 pvec = cross(dir, e2);  
	float  det  = dot(e1, pvec);   

	float inv_det = 1.0/ det;  

	float u = dot(tvec, pvec) * inv_det;  

	if (u < 0.0 || u > 1.0)  
		return vec4(-1,0,0,0);  

	vec3 qvec = cross(tvec, e1);  

	float v = dot(dir, qvec) * inv_det;  

	if (v < 0.0 || (u + v) > 1.0)  
		return vec4(-1,0,0,0);  

	float t = dot(e2, qvec) * inv_det;
	if((index+1) % 2 ==0 ) {
		v = 1.0-v; 
	} else {
		u = 1.0-u;
	} 
	
	normal = normalize(cross(e2,e1));
	return vec4(t,u,v,list_pos.w); //w: material index
	// return vec4(1,1,1,1); //w: material index
}

//pseudorandom number generator
float random(vec3 scale, float seed) {		
	return fract(sin(dot(gl_FragCoord.xyz + seed, scale)) * 43758.5453 + seed);	
}	

//gives a uniform random direction vector
vec3 uniformlyRandomDirection(float seed) {		
	float u = random(vec3(12.9898, 78.233, 151.7182), seed);		
	float v = random(vec3(63.7264, 10.873, 623.6736), seed);		
	float z = 1.0 - 2.0 * u;		
	float r = sqrt(1.0 - z * z);	
	float angle = 6.283185307179586 * v;	
	return vec3(r * cos(angle), r * sin(angle), z);	
}	

//returns a uniformly random vector
vec3 uniformlyRandomVector(float seed) 
{		
	return uniformlyRandomDirection(seed) *  (random(vec3(36.7539, 50.3658, 306.2759), seed));	
}

//function to test if the given ray intersect any object
//if so it returns 0.5 otherwise 1. This darkens the shade
//simulating shadow
float shadow(vec3 origin, vec3 dir ) {
	vec3 tmp;
	for(int i=0;i<int(TRIANGLE_TEXTURE_SIZE);i++) 
	{
		vec4 res = intersectTriangle(origin, dir, i, tmp); 
		if(res.x>0.0 ) { 
		   return 0.5;   
		}
	}
	return 1.0;
}

//function that traces ray with origin and direction from the given light position
vec3 pathtrace(vec3 origin, vec3 ray, vec3 light, float t) {		

	//set the accumulation variable to 0
	//set color mask to 1 and surface colour to background colour
	vec3 colorMask = vec3(1.0);		
	vec3 accumulatedColor = vec3(0.0);
	vec3 surfaceColor=vec3(backgroundColor.xyz);
	
	float diffuse = 1.0;
	// for the total number of bounces
	for(int bounce = 0; bounce < MAX_BOUNCES; bounce++) {			
		//check the ray for intesection with the scene bounding box
		vec2 tNearFar = intersectCube(origin, ray,  aabb);
		
		//if it is outside continue
		if(  tNearFar.x > tNearFar.y)  
		   continue; 
		
		//if it nearest so far set the t paramter
		if(tNearFar.y<t)
			t =   tNearFar.y+1.0;					
		
		vec3 N;
		vec4 val=vec4(t,0,0,0); 

		//brute force check all triangles for intersection with the ray
		for(int i=0;i<int(TRIANGLE_TEXTURE_SIZE);i++) 
		{
			vec3 normal;
			vec4 res = intersectTriangle(origin, ray, i, normal); 
			//if intersection found, store the result and normal
		 	if(res.x>0.001 && res.x <  val.x) { 
			   val = res;   
			   N = normal;
		    }
		}
		   
		// if this is a valid intersection
		if(  val.x < t) {			  	
			//calcualte the surface color 
												// w  : material index
												// yz ; uv coordinates
			// surfaceColor = mix(texture(textureMaps, val.yzw), vec4(1.0), vec4(val.w==255.0) ).xyz; 
			surfaceColor = mix(vec4(0,1,1,1), vec4(1.0), vec4(0) ).xyz; 
			
			//find the hit position, change the ray origin to the new hit position
			//and get a new random ray direction 
			vec3 hit = origin + ray * val.x;	
			origin = hit;	
			ray = uniformlyRandomDirection(time + float(bounce));	
			
			//jitter the light to reduce sampling artifacts
			vec3  jitteredLight  =  light + ray;
			vec3 L = normalize(jitteredLight - hit);			
			
			//calcualte the diffuse component
			diffuse = max(0.0, dot(L, normalize(N) ));			 		
			colorMask *= surfaceColor;			

			//check for shadows
			float inShadow = shadow(hit+ N*0.0001, L);

			//accumulate the colour
			accumulatedColor += colorMask * diffuse * inShadow; 
			t = val.x;
		}  			
	}		
	//if the accumulatedColor is 0, we did not intersect any geometry
	//return the product of the diffuse component and the surface colour
	if(accumulatedColor == vec3(0.0))
		return surfaceColor*diffuse;
	else
		//otherwise divide the accumulated colour with the total number of bounces
		return accumulatedColor/float(float(MAX_BOUNCES)-1.0);	
}	


//function that traces ray with origin and direction from the given light position
vec3 pathtrace(vec3 origin, vec3 ray, vec3 light, float t, float time) {		

	//set the accumulation variable to 0
	//set color mask to 1 and surface colour to background colour
	vec3 colorMask = vec3(1.0);		
	vec3 accumulatedColor = vec3(0.0);
	vec3 surfaceColor=vec3(backgroundColor.xyz);
	
	float diffuse = 1.0;
	// for the total number of bounces
	for(int bounce = 0; bounce < MAX_BOUNCES; bounce++) {			
		//check the ray for intesection with the scene bounding box
		vec2 tNearFar = intersectCube(origin, ray,  aabb);
		
		//if it is outside continue
		if(  tNearFar.x > tNearFar.y)  
		   continue; 
		
		//if it nearest so far set the t paramter
		if(tNearFar.y<t)
			t =   tNearFar.y+1.0;					
		
		vec3 N;
		vec4 val=vec4(t,0,0,0); 

		//brute force check all triangles for intersection with the ray
		for(int i=0;i<int(TRIANGLE_TEXTURE_SIZE);i++) 
		{
			vec3 normal;
			vec4 res = intersectTriangle(origin, ray, i, normal); 
			//if intersection found, store the result and normal
		 	if(res.x>0.001 && res.x <  val.x) { 
			   val = res;   
			   N = normal;
		    }
		}
		   
		// if this is a valid intersection
		if(  val.x < t) {			  	
			//calcualte the surface color 
												// w  : material index
												// yz ; uv coordinates
			// surfaceColor = mix(texture(textureMaps, val.yzw), vec4(1.0), vec4(val.w==255.0) ).xyz; 
			surfaceColor = mix(vec4(0,1,1,1), vec4(1.0), vec4(0) ).xyz; 
			
			//find the hit position, change the ray origin to the new hit position
			//and get a new random ray direction 
			vec3 hit = origin + ray * val.x;	
			origin = hit;	
			ray = uniformlyRandomDirection(time + float(bounce));	
			
			//jitter the light to reduce sampling artifacts
			vec3  jitteredLight  =  light + ray;
			vec3 L = normalize(jitteredLight - hit);			
			
			//calcualte the diffuse component
			diffuse = max(0.0, dot(L, normalize(N) ));			 		
			colorMask *= surfaceColor;			

			//check for shadows
			float inShadow = shadow(hit+ N*0.0001, L);

			//accumulate the colour
			accumulatedColor += colorMask * diffuse * inShadow; 
			t = val.x;
		}  			
	}		
	//if the accumulatedColor is 0, we did not intersect any geometry
	//return the product of the diffuse component and the surface colour
	if(accumulatedColor == vec3(0.0))
		return surfaceColor*diffuse;
	else
		//otherwise divide the accumulated colour with the total number of bounces
		return accumulatedColor/float(float(MAX_BOUNCES)-1.0);	
}	

void main()
{ 
	// // set the maximum t value
	float t = 10000.0;  
	
	//set the fragment colour as the background colour 
	vFragColor = backgroundColor;

	//setup the camera for the given texture coordinate
	setup_camera(vUV);
	
	//check if the ray intersects the scene bounding box 
	vec2 tNearFar = intersectCube(eyeRay.origin, eyeRay.dir,  aabb);

	//if we have a valid intersection
	if(tNearFar.x < tNearFar.y  ) {

		if(spp == 0.0f){
			t = tNearFar.y+1.0; //offset the near intersection to remove the depth artifacts
		  		 
			//trace ray from the light positoin in a random direction		
			vec3 light = light_position + uniformlyRandomVector(time);

			//do path tracing here 
			vFragColor = vec4(pathtrace(eyeRay.origin, eyeRay.dir, light, t), 1.0f);
		}else{
			vec3 color;
        	vec3 outColor = vec3(0.0f);
	        for(int s = 0; s < int(spp); s++){
	        	if (spp != 1.0f){
	        		vec3 light = light_position + uniformlyRandomVector(float(s)/100.0f + 0.01f );
	        		color = pathtrace(eyeRay.origin, eyeRay.dir, light, t, float(s)/100.0f + 0.01f);
	        	}
	        	else{
	        		vec3 light = light_position + uniformlyRandomVector(time);
	        		color = pathtrace(eyeRay.origin, eyeRay.dir, light, t, time);
	        	}

	           outColor.r += color.r/spp;
	           outColor.g += color.g/spp;
	           outColor.b += color.b/spp;
	        }
			vFragColor =  vec4(outColor, 1.0f);
		}
	}

	// vFragColor = vec4(-eyePos.z/10.0, 0, 0, 1);

 	// vFragColor = vec4(texture(triangles_list, vec2((vUV.x+1.0)/2.0,(vUV.y+1.0)/2.0) ).xyz ,1.0);


 	// vec4 abc = invMVP*vec4(light_position, 1.0);
 	// vFragColor = vec4( abc.x/55.0f, 0,0, 1.0);

	// vFragColor = vec4((vUV.x+1.0)/2.0,(vUV.y+1.0)/2.0,0.0,1.0 );
	// vFragColor = vec4(vUV.x, vUV.y, 0, 1);
	// vFragColor = backgroundColor;
}


