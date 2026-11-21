cmake_minimum_required(VERSION 3.16)
project(SheyoOS_v1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(atmosphere_kernel SHARED atmosphere_export.cpp)
target_include_directories(atmosphere_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(atmosphere_kernel PROPERTIES PREFIX "")

if(MSVC)
    target_compile_options(atmosphere_kernel PRIVATE /O2 /fp:fast)
else()
    target_compile_options(atmosphere_kernel PRIVATE -O3 -ffast-math)
endif()