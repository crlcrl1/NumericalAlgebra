cmake_minimum_required(VERSION 3.20)

project(HW5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

# enable avx2 and fast math for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
        add_compile_options(/arch:AVX2 /fp:fast)
    else ()
        add_compile_options(-march=native -ffast-math)
    endif ()
endif ()

# detect lto support
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_support OUTPUT lto_error)

add_executable(HW5 main.cpp
        util.h
        util.cpp
        multi_grid.cpp
        multi_grid.h
        ldl.cpp
        ldl.h
        gauss_seidel.cpp
        gauss_seidel.h
        conjugate_gradient.cpp
        conjugate_gradient.h)

target_link_libraries(HW5 Eigen3::Eigen)


# enable lto if supported
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "IPO/LTO is supported: ${lto_support}")
    if (lto_support)
        set_property(TARGET HW5 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
else ()
    message(STATUS "IPO/LTO will not be enabled in non-Release mode")
endif ()