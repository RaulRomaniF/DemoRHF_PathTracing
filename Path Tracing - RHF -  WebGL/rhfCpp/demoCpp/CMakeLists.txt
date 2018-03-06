cmake_minimum_required(VERSION 3.5)

message( "Building tutorial 5 - linux" )
message( "GENERATING MAKEFILES/PROJECTS FOR LINUX" )

find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )
find_package( X11 REQUIRED )
find_package( Threads REQUIRED )
find_package( PkgConfig REQUIRED )
find_package( OpenCV REQUIRED )

pkg_search_module( GLFW REQUIRED glfw3 )

# find_package(assimp REQUIRED)  
#find_package(assimp REQUIRED)
#message( "{OpenCV_LIBS}" )
#message( ${OpenCV_LIBS} )



# c++11
set(CMAKE_CXX_STANDARD 11)

include_directories( /usr/local/include/OpenEXR )
include_directories( /usr/include/OpenEXR/ )
include_directories( /usr/local/include )
link_directories( /usr/local/lib )


add_executable( app
                main.cpp
                # io_exr.cpp
                # libdenoising.cpp
                # libauxiliar.cpp
                # /usr/local/include/glad.c 
                )





# target_link_libraries(MyProject assimp)
target_link_libraries( app
	                   ${CMAKE_THREAD_LIBS_INIT}
	                   ${OpenCV_LIBS}
	                   ${X11_LIBRARIES} 
	                   IlmImf
                       Half 
                       Iex${ILMBASE_LIBSUFFIX}
				       Imath${ILMBASE_LIBSUFFIX}
				       IlmThread${ILMBASE_LIBSUFFIX}
				       ${PTHREAD_LIB} 
				       ${ZLIB_LIBRARIES}	                   

                       #opengl and opencv
				       ${OPENGL_LIBRARIES} 
                       ${GLFW_LIBRARIES} 
                       GLEW::GLEW 
                       SOIL
                       GL
                       GLU
                       glut
                       ${X11_LIBRARIES} 
                       ${CMAKE_THREAD_LIBS_INIT}
                       dl 
                       ${OpenCV_LIBS}
                      )










