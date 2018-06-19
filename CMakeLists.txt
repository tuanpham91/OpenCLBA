cmake_minimum_required(VERSION 3.5)

project(OpenCLBA-Prod)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set (CMAKE_CXX_STANDARD 14)

find_package(VTK 8.1.0 REQUIRED)
find_package(OpenCV REQUIRED )
find_package(PCL 1.8 REQUIRED )
find_package(OpenCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OpenCL_LIBRARY})
