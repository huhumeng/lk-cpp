cmake_minimum_required(VERSION 3.2)
project(lk20-cpp)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

include_directories(
    include
    /usr/local/include/eigen3
)

find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_core
    src/image_process.cc
    src/affine_estimate.cc
)
target_link_libraries(${PROJECT_NAME}_core
    ${OpenCV_LIBS}
)

add_executable(ex_affine example/ex_affine.cc)
target_link_libraries(ex_affine ${PROJECT_NAME}_core)