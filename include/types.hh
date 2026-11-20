#ifndef TYPES_HH
#define TYPES_HH

#include <vector>
#include <string>
#include <unordered_map>
#include <random>


/* --------------
 * Word tokenizer
 * -------------- */
class word_tokenizer_t {
    public:
        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer IDs and vice versa */
        std::unordered_map<std::string, size_t> vocab_token2id;  // Fast (O(1)) for token lookup in word_tokenizer_t::encode()
        std::unordered_map<size_t, std::string> vocab_id2token;  // Fast (O(1)) for ID    lookup in word_tokenizer_t::decode()

        // Constructor
        word_tokenizer_t(const std::string &training_text);

        // Encode (token-to-ID) method
        std::vector<size_t> encode(const std::string &text);

        // Decode (ID-to-token) method
        std::string decode(const std::vector<size_t> &ids);

    private:
        // 'Unknown' and 'end-of-text' token-ID pairs
        std::pair<std::string, size_t> unk, eot;
};


/* ----------------------------------
 * Byte-pair encoding (BPE) tokenizer
 * ---------------------------------- */
class bpe_tokenizer_t {
    public:
        /* Token-to-ID and ID-to-token vocabularies, i.e., hash maps relating
         * unique tokens from a training text to integer IDs and vice versa */
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

    private:
        // 'Unknown' and 'end-of-text' token-ID pairs
        std::pair<std::string, size_t> unk, eot;

        // 'End-of-word' token
        std::string eow;
};


/* -------------------
 * Model (transformer)
 * ------------------- */
class transformer_t {
    public:
        // Constructor
        transformer_t(const size_t &nv,  // Vocabulary size
                      std::mt19937 &gen);

        // Forward method
        void forward(const std::vector<size_t> &ids,
                     const size_t &start_id_train);

        // Backward method
        double backward(const std::vector<size_t> &ids,
                        const size_t &start_id_train);

    private:
        // ***** Internal parameters *****
        size_t nv;
        std::mt19937 gen;
        std::uniform_real_distribution<double> udist;


        // ***** Learnable parameters *****
        std::vector<double> vocab_embeddings;
        std::vector<double>   pos_embeddings;

        std::vector<double> scale1, scale2, scale3;
        std::vector<double> shift1, shift2, shift3;

        std::vector<double> Wq, Wk, Wv;

        std::vector<double> ffn_W1, ffn_W2;
        std::vector<double> ffn_b1, ffn_b2;

        std::vector<double> logits_W;  // Empty if weight tying is active
        std::vector<double> logits_b;


        // ***** Gradients *****
        std::vector<double> d_vocab_embeddings;
        std::vector<double>   d_pos_embeddings;

        std::vector<double> d_scale1, d_scale2, d_scale3;
        std::vector<double> d_shift1, d_shift2, d_shift3;

        std::vector<double> dWq, dWk, dWv;

        std::vector<double> d_ffn_W1, d_ffn_W2;
        std::vector<double> d_ffn_b1, d_ffn_b2;

        std::vector<double> d_logits_W;  // Empty if weight tying is active
        std::vector<double> d_logits_b;


        // ***** Forward workspace *****
        std::vector<double> x, x1, x2;
        std::vector<double> queries, keys, values;
        std::vector<double> attention, context;
        std::vector<double> xhat1, xhat2, xhat3;
        std::vector<double> sigmas_inv1, sigmas_inv2, sigmas_inv3;
        std::vector<double> ffn_h, ffn_h_prime, ffn_out;
        std::vector<double> logits;

        // ***** Backward workspace *****
        std::vector<double> probs, dx, dz, dA;
        std::vector<double> dQ, dK, dV;
};

#endif
