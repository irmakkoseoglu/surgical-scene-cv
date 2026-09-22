cmake_minimum_required(VERSION 3.16)
project(surgical_scene_cv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d dnn)

foreach(tool segment stereo_calibrate stereo_depth)
  add_executable(${tool} src/${tool}.cpp)
  target_include_directories(${tool} PRIVATE include)
  target_link_libraries(${tool} PRIVATE ${OpenCV_LIBS})
  target_compile_options(${tool} PRIVATE -Wall -Wextra)
endforeach()
