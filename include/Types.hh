#ifndef TYPES_HH
#define TYPES_HH

#include <vector>
#include <string>
#include <unordered_map>


/* --------------
 * Word tokenizer
 * -------------- */
class tokenizer_t {
    public:
        // 'Unknown' and 'end-of-text' tokens
        std::pair<std::string, size_t> unk, eot;

        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer IDs and vice versa     */
        std::unordered_map<std::string, size_t> vocab_token2id;  // Fast (O(1)) for token lookup in tokenizer_t::encode()
        std::unordered_map<size_t, std::string> vocab_id2token;  // Fast (O(1)) for ID    lookup in tokenizer_t::decode()

        // Constructor
        tokenizer_t(const std::string &training_text);

        // Encode (token-to-ID) method
        std::vector<size_t> encode(const std::string &text);

        // Decode (ID-to-token) method
        std::string decode(const std::vector<size_t> &ids);
};


/* ----------------------------------
 * Byte-pair encoding (BPE) tokenizer
 * ---------------------------------- */
class tokenizer_bpe_t {
    public:
        // 'Unknown' and 'end-of-text' tokens
        std::pair<std::string, size_t> unk, eot;

        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer IDs and vice versa     */
        std::unordered_map<std::string, size_t> vocab_token2id;
        std::unordered_map<size_t, std::string> vocab_id2token;

        // Constructor
        tokenizer_bpe_t(const std::string &training_text,
                        const size_t      &nmerges);
};


#endif
