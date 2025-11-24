#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <regex>
#include <algorithm>

#include "Types.hh"
#include "Parameters.hh"

using namespace std;


/* ==========================================================================
 * Constructor building the vocabulary, i.e., a hash map relating unique text
 * tokens to integer IDs
 * ========================================================================== */
tokenizer_t::tokenizer_t(const string &training_text) {
    // This regex matches words, numbers, and punctuation as individual tokens
    regex                 token_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator tokens_it(training_text.begin(), training_text.end(), token_re);
    sregex_token_iterator end;

    size_t id = 0;

    while (tokens_it != end) {
        /* Make all tokens lowercase to avoid duplicating them if they occur
         * multiple times with different cases                                  */
        auto token = tokens_it->str();
        transform(token.begin(), token.end(), token.begin(), //::tolower);
                  /* Wrap ::tolower in a lambda to cast the input into
                   * unsigned char to avoid undefined behavior if the input
                   * char is signed and negative (non-ASCII byte)           */
                  [](unsigned char c)->char {
                      return static_cast<char>(std::tolower(c));
                  }
                 );

        /* Try inserting the token into the token-to-ID vocabulary (O(1) for
         * token lookup in tokenizer_t::encode()); this will only succeed if the
         * token is not in the vocabulary yet, since the tokens are the keys in
         * the map and keys are unique                                          */
        if ((this->vocab_token2id).emplace(token, id).second) {
            ++id;
        } else {
            #if (VERBOSE)
            cout << "Skipping repeated token '" << token
                 << "' while creating the token-to-ID vocabulary" << endl;
            #endif
        }

        ++tokens_it;
    }


    /* Add extra ids to handle unknown and end-of-text (useful when training
     * with multiple text sources) tokens                                       */
    const auto original_vocab_size = (this->vocab_token2id).size();
    (this->unk) = {"<|unknown|>",     original_vocab_size};
    (this->eot) = {"<|end-of-text|>", original_vocab_size + 1};

    if (not (this->vocab_token2id).emplace(this->unk).second) {
        throw runtime_error("Insertion of 'unknown' token failed");
    }

    if (not (this->vocab_token2id).emplace(this->eot).second) {
        throw runtime_error("Insertion of 'end-of-text' token failed");
    }


    /* Build the ID-to-token vocabulary for fast (O(1)) ID lookup in
     * tokenizer_t::decode()                                                      */
    (this->vocab_id2token).reserve((this->vocab_token2id).size());

    for (const auto &[token, id] : (this->vocab_token2id)) {
        if (not (this->vocab_id2token).emplace(id, token).second) {
            ostringstream exception_ss;
            exception_ss << "Unexpected repetition of element [" << id << ", " << token
                         << "] in the ID-to-token vocabulary. This may happen if ID " << id
                         << " is not unique in the token-to-ID vocabulary.";
            throw runtime_error(exception_ss.str());
        }
    }
}



/* ============================================================================
 * Encode method using the token-to-ID vocabulary to convert an input text into
 * the corresponding set of token IDs
 * ============================================================================ */
vector<size_t> tokenizer_t::encode(const string &text) {
    // This regex matches words, numbers, and punctuation as individual tokens
    regex                 token_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator tokens_it(text.begin(), text.end(), token_re);
    sregex_token_iterator end;

    const auto nmatches = static_cast<size_t>(distance(tokens_it, end));
    vector<size_t> tokenIDs(nmatches);

    size_t i = 0;
    for (sregex_token_iterator it(text.begin(), text.end(), token_re); it != end; ++it, ++i) {
        /* Make all tokens lowercase to avoid duplicating them if the same
         * word appears multiple times with different cases                 */
        auto token = it->str();
        transform(token.begin(), token.end(), token.begin(), //::tolower);
                  /* Wrap ::tolower in a lambda to cast the input into
                   * unsigned char to avoid undefined behavior if the input
                   * char is signed and negative (non-ASCII byte)           */
                  [](unsigned char c)->char {
                      return static_cast<char>(std::tolower(c));
                  }
                 );

        /* Convert the token into the ID if found in the vocabulary, otherwise
         * set the ID t 'unknown'                                               */
        try {
            /* The at() method on the RHS throws an exception if the token is
             * not in the token-to-ID vocabulary                                */
            tokenIDs.at(i) = (this->vocab_token2id).at(token);
        } catch (exception &e) {
            tokenIDs.at(i) = (this->unk).second;
            cerr << "Unknown token '" << token << "': setting ID to 'unknown' token ID "
                 << (this->unk).second << " (exception: \"" << e.what() << "\")" << endl;
        }
    }

    return tokenIDs;
}



/* ========================================================================
 * Decode method using the ID-to-token vocabulary to convert a set of input
 * token IDs into the corresponding text tokens
 * ========================================================================= */
string tokenizer_t::decode(const vector<size_t> &ids) {
    ostringstream decoded_text_ss;

    for (const auto &id : ids) {
        try {
            decoded_text_ss << (this->vocab_id2token).at(id) << " ";  // NOTE: extra space to separate tokens
        } catch (exception &e) {
            ostringstream exception_ss;
            exception_ss << "Unknown token ID " << id
                         << ": this should never happen because the 'unknown' token should be part of the dictionary. Please check the code's correctness (exception: \""
                         << e.what() << "\")";
            throw runtime_error(exception_ss.str());
        }
    }

    return decoded_text_ss.str();
}
