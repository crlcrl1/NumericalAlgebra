cmake_minimum_required(VERSION 3.20)

project(Final LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${CMAKE_SOURCE_DIR}/include")

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# enable avx2 and fast math for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
        add_compile_options(/arch:AVX2 /fp:fast)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        add_compile_options(-xHost -march=native -ffast-math -mtune=native -ftree-vectorize)
    else ()
        add_compile_options(-march=native -ffast-math -mtune=native -ftree-vectorize)
    endif ()
endif ()

# detect lto support
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_support OUTPUT lto_error)


# enable lto if supported
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "IPO/LTO is supported: ${lto_support}")
    if (lto_support)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif ()
else ()
    message(STATUS "IPO/LTO will not be enabled in non-Release mode")
endif ()

add_executable(Final main.cpp
        util.cpp
        util.h
        v_cycle.cpp
        v_cycle.h
        uzawa.cpp
        uzawa.h
        equation.cpp
        equation.h
)

target_link_libraries(Final Eigen3::Eigen)
target_link_libraries(Final OpenMP::OpenMP_CXX)