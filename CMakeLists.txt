cmake_minimum_required(VERSION 4.3)
project(RidgeRegression)

set(CMAKE_CXX_STANDARD 23)

add_executable(RidgeRegression main.cpp
        math_vector.h
        matrix.h)
