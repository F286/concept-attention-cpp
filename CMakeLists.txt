cmake_minimum_required(VERSION 3.16)
project(concept_attention_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(mini_torch
    src/tensor.cpp
    src/linear.cpp
    src/attention.cpp
    src/model.cpp
)

target_include_directories(mini_torch PUBLIC include)

enable_testing()

add_executable(all_tests tests/all_tests.cpp)
target_link_libraries(all_tests PRIVATE mini_torch)
add_test(NAME all_tests COMMAND all_tests)
