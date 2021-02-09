cmake_minimum_required(VERSION 3.2)

set (CMAKE_CXX_STANDARD 11)

project(opencv_practice)

file(COPY ${CMAKE_SOURCE_DIR}/res DESTINATION ${CMAKE_BINARY_DIR})

# add_subdirectory(thirdparty)
add_subdirectory(res)
add_subdirectory(src)
