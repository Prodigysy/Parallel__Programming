cmake_minimum_required(VERSION 3.11)

project(Matrix CXX)

set(CMAKE_CXX_STANDARD 23)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true)

    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
    set(CMAKE_CXX_FLAGS " /Ob0 ${CMAKE_CXX_FLAGS} ")
endif()

add_library(Matrix SHARED "include/Matrix/Matrix.h" "src/Matrix.cc")
target_include_directories(Matrix PUBLIC "include")

add_executable(MatrixExe "tests/MatrixTests.cc")
target_link_libraries(MatrixExe Matrix)