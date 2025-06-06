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

add_executable(genesis_model src/genesis_model.cpp)
target_link_libraries(genesis_model PRIVATE mini_torch)

add_executable(baseline_model src/baseline_main.cpp)
target_link_libraries(baseline_model PRIVATE mini_torch)

add_executable(train_models src/train_models.cpp)
target_link_libraries(train_models PRIVATE mini_torch)

enable_testing()
add_executable(tensor_tests tests/tensor_tests.cpp)
target_link_libraries(tensor_tests PRIVATE mini_torch)
add_test(NAME tensor_tests COMMAND tensor_tests)

add_executable(attention_tests tests/attention_tests.cpp)
target_link_libraries(attention_tests PRIVATE mini_torch)
add_test(NAME attention_tests COMMAND attention_tests)
