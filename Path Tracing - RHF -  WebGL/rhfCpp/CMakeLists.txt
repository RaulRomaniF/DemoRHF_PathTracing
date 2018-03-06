cmake_minimum_required(VERSION 3.5)

message( "Building tutorial 5 - linux" )
message( "GENERATING MAKEFILES/PROJECTS FOR LINUX" )

find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )
find_package( X11 REQUIRED )
find_package( Threads REQUIRED )
find_package( PkgConfig REQUIRED )
find_package( OpenCV REQUIRED )
find_package( CUDA  )

pkg_search_module( GLFW REQUIRED glfw3 )

# find_package(assimp REQUIRED)  
#find_package(assimp REQUIRED)
#message( "{OpenCV_LIBS}" )
#message( ${OpenCV_LIBS} )

# c++11
set(CMAKE_CXX_STANDARD 11)

# ------ Cuda 
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
# link_directories(${CUDA_LIBRARY_DIRS})
# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_ARCH -gencode arch=compute_35,code=sm_35)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -g -G --std=c++11)


# ------ OpenMP support required 
find_package(OpenMP REQUIRED)
include_directories(${OPENMP_INCLUDE_DIRS})
link_directories(${OPENMP_LIBRARY_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# ------ End OpenMP support required 


include_directories( /usr/local/include/OpenEXR )
include_directories( /usr/include/OpenEXR/ )
include_directories( /usr/local/include )
link_directories( /usr/local/lib )


# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)

# cuda_compile(FOO_O cudaBase.cu )

set(__cuda_arch_bin "3.0 3.5" )
# add_executable( main.out 
cuda_add_executable( main.out 
                rhf.cpp
                io_exr.cpp
                cudaBase.cu

                # ${FOO_O}
                libdenoising.cpp
                libauxiliar.cpp
                io_png.cpp
                # pathTracer.cpp
                # src/Grid.cpp
                # src/RenderableObject.cpp
                # src/GLSLShader.cpp
                # Object.cpp
                # model.cpp
                
                )





# target_link_libraries(MyProject assimp)
target_link_libraries( main.out
	                   ${CMAKE_THREAD_LIBS_INIT}
	                   ${OpenCV_LIBS}
	                   ${X11_LIBRARIES} 
	                   IlmImf
                       Half 
                       png
                       curl
                       Iex${ILMBASE_LIBSUFFIX}
        				       Imath${ILMBASE_LIBSUFFIX}
        				       IlmThread${ILMBASE_LIBSUFFIX}
        				       ${PTHREAD_LIB} 
        				       ${ZLIB_LIBRARIES}	  
                       SOIL                 

                      )










