cmake_minimum_required(VERSION 3.16)
project(concept_attention_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(external/catch2)

add_library(catch2_obj OBJECT external/catch2/catch.cpp)
target_compile_definitions(catch2_obj PRIVATE CATCH_AMALGAMATED_CUSTOM_MAIN)

add_library(mini_torch
    src/tensor.cpp
    src/linear.cpp
    src/attention.cpp
    src/model.cpp
)

target_include_directories(mini_torch PUBLIC include)

add_executable(concept_app
    src/main.cpp
    src/genesis_model.cpp
    src/baseline_main.cpp
    src/train_models.cpp
    tests/tensor_tests.cpp
    tests/attention_tests.cpp
    $<TARGET_OBJECTS:catch2_obj>
)
target_link_libraries(concept_app PRIVATE mini_torch)
enable_testing()
add_test(NAME all_tests COMMAND concept_app test)
