cmake_minimum_required(VERSION 2.8)
project(PersonDetector)


find_package(OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(CUDA REQUIRED )

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

#SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -stdlib=libstdc++" )

set( NAME_SRC
    src/PersonDetector.cpp    
)

CUDA_add_executable(PersonDetector src/PersonDetector.cpp)
target_link_libraries(PersonDetector ${OpenCV_LIBS})
target_link_libraries(PersonDetector ${Boost_LIBRARIES}  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY} )
