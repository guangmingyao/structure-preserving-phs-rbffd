cmake_minimum_required(VERSION 3.16)
project(SchrodingerPHS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(schrodinger_experiments src/main.cpp)
target_compile_options(schrodinger_experiments PRIVATE -O3)

add_executable(schrodinger_higher_grid src/higher_grid.cpp)
target_compile_options(schrodinger_higher_grid PRIVATE -O3)
