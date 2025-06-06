#include "mini_torch/model.h"
#include <iostream>
#include <string>

#define CATCH_CONFIG_RUNNER
#include "catch_amalgamated.hpp"

/// @brief Demonstration of baseline model
int run_baseline_demo();
/// @brief Demonstration of genesis model
int run_genesis_demo();
/// @brief Training demo comparing models
int run_training_demo();

/// @brief Application entry with subcommand dispatch and test support
int main(int argc, char** argv) {
    if (argc > 1) {
        std::string cmd = argv[1];
        if (cmd == "baseline") return run_baseline_demo();
        if (cmd == "genesis") return run_genesis_demo();
        if (cmd == "train") return run_training_demo();
        if (cmd == "test") {
            Catch::Session session;
            return session.run(argc - 1, argv + 1);
        }
    }
    std::cout << "Usage: " << argv[0] << " [baseline|genesis|train|test]\n";
    return 0;
}

