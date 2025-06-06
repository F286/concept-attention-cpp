#include "mini_torch/tokenizer.h"
#include <cctype>
#include <sstream>

std::vector<std::string> Tokenizer::operator()(const std::string &text) const {
    std::string cleaned;
    cleaned.reserve(text.size());
    for(char ch : text){
        char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if(std::isalnum(static_cast<unsigned char>(lower)))
            cleaned.push_back(lower);
        else
            cleaned.push_back(' ');
    }
    std::vector<std::string> tokens;
    std::stringstream ss(cleaned);
    std::string tok;
    while(ss >> tok)
        tokens.push_back(tok);
    return tokens;
}
