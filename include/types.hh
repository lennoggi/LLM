#ifndef TYPES_HH
#define TYPES_HH

#include <vector>
#include <string>
#include <unordered_map>


/* --------------
 * Word tokenizer
 * -------------- */
class word_tokenizer_t {
    private:
        // 'Unknown' and 'end-of-text' token-ID pairs
        std::pair<std::string, size_t> unk, eot;

    public:
        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer IDs and vice versa     */
        std::unordered_map<std::string, size_t> vocab_token2id;  // Fast (O(1)) for token lookup in word_tokenizer_t::encode()
        std::unordered_map<size_t, std::string> vocab_id2token;  // Fast (O(1)) for ID    lookup in word_tokenizer_t::decode()

        // Constructor
        word_tokenizer_t(const std::string &training_text);

        // Encode (token-to-ID) method
        std::vector<size_t> encode(const std::string &text);

        // Decode (ID-to-token) method
        std::string decode(const std::vector<size_t> &ids);
};


/* ----------------------------------
 * Byte-pair encoding (BPE) tokenizer
 * ---------------------------------- */
class bpe_tokenizer_t {
    private:
        // 'Unknown' and 'end-of-text' token-ID pairs
        std::pair<std::string, size_t> unk, eot;

        // 'End-of-word' token
        std::string eow;

    public:
        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer IDs and vice versa     */
        std::unordered_map<std::string, size_t> vocab_token2id;
        std::unordered_map<size_t, std::string> vocab_id2token;

        // Constructor
        bpe_tokenizer_t(const std::string &training_text,
                        const std::string &eow,
                        const size_t      &max_vocab_size);

        // Encode (token-to-ID) method
        std::vector<size_t> encode(const std::string &text);

        // Decode (ID-to-token) method
        std::string decode(const std::vector<size_t> &ids);
};


#endif
