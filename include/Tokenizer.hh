#ifndef TOKENIZER_HH
#define TOKENIZER_HH

#include <vector>
#include <string>
#include <unordered_map>


class tokenizer_t {
    public:
        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer unique IDs and vice
         * versa                                                                */
        std::unordered_map<std::string, int> vocab_token2id;  // Fast (O(1)) for token lookup in tokenizer_t::encode()
        std::unordered_map<int, std::string> vocab_id2token;  // Fast (O(1)) for ID    lookup in tokenizer_t::decode()

        // Constructor
        tokenizer_t(const std::string &training_text);

        // Encode (token-to-ID) method
        std::vector<int> encode(const std::string &text);

        // Decode (ID-to-token) method
        std::string decode(const std::vector<int> &ids);
};


#endif
