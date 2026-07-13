cmake_minimum_required(VERSION 3.15)
project(matrix_op CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi> $<$<CONFIG:Release>:/GL> $<$<CONFIG:Release>:/fp:fast>)
    add_link_options($<$<CONFIG:Release>:/LTCG>)
else()
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native>)
endif()

include_directories(include)

add_library(matrix_op_lib
    src/matrix_utils.cpp
    src/transpose.cpp
    src/multiply.cpp
)

add_executable(matrix_bench src/main.cpp)
target_link_libraries(matrix_bench PRIVATE matrix_op_lib)

add_executable(matrix_tests
    tests/test_main.cpp
    tests/test_matrix_utils.cpp
    tests/test_timer.cpp
    tests/test_transpose.cpp
    tests/test_multiply.cpp
)
target_include_directories(matrix_tests PRIVATE tests)
target_link_libraries(matrix_tests PRIVATE matrix_op_lib)

enable_testing()
add_test(NAME unit_tests COMMAND matrix_tests)
