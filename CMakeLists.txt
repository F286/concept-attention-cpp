cmake_minimum_required(VERSION 3.16)
project(concept_attention_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(mini_torch
    src/tensor.cpp
    src/linear.cpp
    src/attention.cpp
    src/loss.cpp
    src/optim.cpp
    src/model.cpp
)

target_include_directories(mini_torch PUBLIC include)

enable_testing()

add_executable(all_tests
    tests/main.cpp
    tests/tensor_tests.cpp
    tests/linear_tests.cpp
    tests/loss_tests.cpp
    tests/optim_tests.cpp
    tests/attention_tests.cpp
    tests/model_tests.cpp
    tests/train_run_tests.cpp
)
target_link_libraries(all_tests PRIVATE mini_torch)
add_test(NAME all_tests COMMAND all_tests "~[TRAIN]")

add_test(NAME training COMMAND all_tests "[TRAIN]")
set_tests_properties(training PROPERTIES LABELS "TRAIN")
