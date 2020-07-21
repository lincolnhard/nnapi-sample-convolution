cmake_minimum_required(VERSION 3.4.1)

project(nnapi-sample-convolution)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

find_library(neuralnetworks-lib
		neuralnetworks)

find_library(android-lib
		android)

add_executable(${PROJECT_NAME}
			simple_model.cpp
			main.cpp)

target_link_libraries(${PROJECT_NAME}
				${neuralnetworks-lib}
				${android-lib})
