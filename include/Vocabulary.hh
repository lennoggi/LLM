#ifndef VOCABULARY_HH
#define VOCABULARY_HH

#include <vector>
#include <string>
#include <unordered_map>


class vocabulary_t {
    public:
        std::vector<std::string> tokens;
        std::unordered_map<std::string, size_t> tokens_hashmap;
        vocabulary_t(const std::string &text);
};


#endif
