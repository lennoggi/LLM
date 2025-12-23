#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <regex>
#include <algorithm>

#include "Types.hh"
#include "Parameters.hh"

using namespace std;


/* =================================================================
 * Constructor building the token-to-ID and ID-to-token vocabularies
 * ================================================================= */
bpe_tokenizer_t::bpe_tokenizer_t(const string &training_text,
                                 const string &eow,
                                 const size_t &max_vocab_size) {
    /* -------------------------------------------------------------------
     * 1. Build a list of all the words in the training text as vectors of
     *    individual characters
     * -------------------------------------------------------------------      */
    /* This regex matches words, numbers, and punctuation as individual tokens,
     * which should be regarded as lists of "symbols"                           */
    regex                 word_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator words_it(training_text.begin(), training_text.end(), word_re);
    sregex_token_iterator end;

    vector<vector<string>> words_list;

    while (words_it != end) {
        /* Make all words lowercase to avoid duplicating them if they occur
         * multiple times with different cases                                  */
        auto word = words_it->str();
        transform(word.begin(), word.end(), word.begin(), //std::tolower);
                  /* Wrap ::tolower in a lambda to cast the input into
                   * unsigned char to avoid undefined behavior if the input
                   * char is signed and negative (non-ASCII byte)           */
                  [](unsigned char c)->char {
                      return static_cast<char>(std::tolower(c));
                  }
                 );

        /* Convert the word into a vector of strings (for now just individual
         * characters, then these will be iteratively replaced with strings of
         * multiple characters)                                                 */
        const auto word_size = word.size();
        vector<string> word_strvec(word_size);

        for (auto i = decltype(word_size){0}; i < word_size; ++i) {
            word_strvec.at(i) = word.at(i);
        }

        // Append the end-of-word character to the last character in the word
        word_strvec.at(word_size - 1) += eow;

        // Insert the word into the word list if not there yet
        if (find(words_list.begin(), words_list.end(), word_strvec) == words_list.end()) {
            words_list.emplace_back(word_strvec);
        }

        ++words_it;
    }


    /* --------------------------------------------------------------
     * 2. Add each individual character to the token-to-ID vocabulary
     * -------------------------------------------------------------- */
    size_t id = 0;

    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
            // Sanity check
            if (token == eow) {
                throw runtime_error("bpe_tokenizer_t(): found end-of-word character token in the training text. This is not supported: please change the end-of-word character to something not present in the training text.");
            }

            /* Try inserting the token into the token-to-ID vocabulary (O(1) for
             * token lookup in bpe_tokenizer_t::encode()); this will only
             * succeed if the token is not in the vocabulary yet, since the
             * tokens are the keys in the map and keys are unique               */
            if ((this->vocab_token2id).emplace(token, id).second) {
                ++id;
            } else {
                #if (VERBOSE)
                cout << "Skipping repeated token '" << token
                     << "' while creating the token-to-ID vocabulary" << endl;
                #endif
            }
        }
    }

    if ((this->vocab_token2id).size() > max_vocab_size) {
        ostringstream error_ss;
        error_ss << "bpe_tokenizer_t(): initial vocabulary size (" << (this->vocab_token2id).size()
                 << ") larger than maximum allowed vocabulary size (" << max_vocab_size
                 << "). Please increase the maximum allowed vocabulary size to at least "
                 << (this->vocab_token2id).size() << ".";
        throw runtime_error(error_ss.str());
    }


    /* ----------------------------------------------------------------------
     * 3. For each word in the training text, loop over all symbol pairs
     *    (initially symbol=character, then characters will be iteratively
     *    merged into subwords) in the word and build a temporary map of
     *    symbol-pairs->frequency. Then, find the symbol pair that occurs the
     *    most, replace its separate symbols in the words list with that
     *    symbol pair (a single string), and add the merged pair to the
     *    token-to-ID vocabulary. Repeat until the token-to-ID vocabulary
     *    reaches the maximum, user-specified, size.
     * ----------------------------------------------------------------------   */
    const auto nwords = words_list.size();

    if (nwords < 1) {
        throw runtime_error("bpe_tokenizer_t(): need at least one word in the text");
    }

    while ((this->vocab_token2id).size() < max_vocab_size) {
        unordered_map<string, size_t> symbolpairs_freq;

        for (const auto &word_strvec : words_list) {
            for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size()-1; ++i) {
                const string symbol_pair = word_strvec.at(i) + word_strvec.at(i+1);

                /* Try adding 'symbol_pair' to 'symbolpairs_freq' with an
                 * initial frequency of 1; if that fails, that means the symbol
                 * pair is already in the map, so increment its frequency by 1. */
                if (not symbolpairs_freq.emplace(symbol_pair, 1).second) {
                    try {
                        symbolpairs_freq.at(symbol_pair) += 1;
                    } catch (const exception &e) {
                        ostringstream exception_ss;
                        exception_ss << "bpe_tokenizer_t(): failed to access the symbol-pair->frequency map at key '" << symbol_pair <<
                            "'. This key should exist, please check the code (exception: \"" << e.what() << "\").";
                        throw runtime_error(exception_ss.str());
                    }
                }
            }
        }

        const auto iterator_maxcounts = max_element(symbolpairs_freq.begin(), symbolpairs_freq.end(),
                                                    [](const auto &a, const auto &b){
                                                        return (a.second < b.second);
                                                    });
        const auto &most_common_symbolpair      = iterator_maxcounts->first;
        const auto &most_common_symbolpair_freq = iterator_maxcounts->second;

        assert(most_common_symbolpair_freq > 0);

        if (most_common_symbolpair_freq == 1) {
            cerr << "********** WARNING **********" << endl
                 << "Merging symbols stopped at vocabulary size " << (this->vocab_token2id).size()
                 << " (maximum allowed vocabulary size: " << max_vocab_size
                 << ") because no other merges are possible" << endl
                 << "*****************************" << endl;
            break;
        } else {
            // Update the words list for the next iteration
            for (auto &word_strvec : words_list) {
                for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size()-1; ++i) {
                    const string symbol_pair = word_strvec.at(i) + word_strvec.at(i+1);
                    if (symbol_pair == most_common_symbolpair) {
                        word_strvec.at(i)   = most_common_symbolpair;
                        word_strvec.at(i+1) = "";
                    }
                }

                /* Delete empty strings (empty because they have been merged above)
                 * from the "word" (list of symbols)                                */
                for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size(); ++i) {
                    if (word_strvec.at(i) == "") {
                        word_strvec.erase(word_strvec.begin() + i);
                    }
                }
            }
        }

        // Add the most common symbol pair to the token-to-ID vocabulary
        if ((this->vocab_token2id).emplace(most_common_symbolpair, id).second) {
            ++id;
        } else {
            #if (VERBOSE)
            cout << "Skipping repeated token '" << most_common_symbolpair 
                 << "' while creating the token-to-ID vocabulary" << endl;
            #endif
        }
    }


    /* ----------
     * 4. Wrap up
     * ---------- */
    /* Add extra IDs to the token-to-ID vocabulary to handle unknown and
     * end-of-text (useful when training with multiple text sources) tokens     */
    const auto original_vocab_size = (this->vocab_token2id).size();
    (this->unk) = {"<|unknown|>",     original_vocab_size};
    (this->eot) = {"<|end-of-text|>", original_vocab_size + 1};

    if (not (this->vocab_token2id).emplace(this->unk).second) {
        throw runtime_error("bpe_tokenizer_t(): insertion of 'unknown' token failed");
    }

    if (not (this->vocab_token2id).emplace(this->eot).second) {
        throw runtime_error("bpe_tokenizer_t(): insertion of 'end-of-text' token failed");
    }

    /* Build the ID-to-token vocabulary for fast (O(1)) ID lookup in
     * bpe_tokenizer_t::decode()                                                */
    (this->vocab_id2token).reserve((this->vocab_token2id).size());

    for (const auto &[token, id] : (this->vocab_token2id)) {
        if (not (this->vocab_id2token).emplace(id, token).second) {
            ostringstream exception_ss;
            exception_ss << "bpe_tokenizer_t(): unexpected repetition of element [" << id << ", " << token
                         << "] in the ID-to-token vocabulary. This may happen if ID " << id
                         << " is not unique in the token-to-ID vocabulary.";
            throw runtime_error(exception_ss.str());
        }
    }

    // Initialize the internal end-of-word string for the encode() method
    (this->eow) = eow;
}



/* ============================================================================
 * Encode method using the token-to-ID vocabulary to convert an input text into
 * the corresponding set of token IDs
 * ============================================================================ */
vector<size_t> bpe_tokenizer_t::encode(const string &text) {
    /* ---------------------------------------------------------------------
     * 1. Build a list of all the words in the text as vectors of individual
     *    characters
     * ---------------------------------------------------------------------    */
    /* This regex matches words, numbers, and punctuation as individual tokens,
     * which should be regarded as lists of "symbols"                           */
    regex                 word_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator words_it(text.begin(), text.end(), word_re);
    sregex_token_iterator end;

    vector<vector<string>> words_list;

    while (words_it != end) {
        /* Make all words lowercase to avoid duplicating them if they occur
         * multiple times with different cases                                  */
        auto word = words_it->str();
        transform(word.begin(), word.end(), word.begin(), //std::tolower);
                  /* Wrap ::tolower in a lambda to cast the input into
                   * unsigned char to avoid undefined behavior if the input
                   * char is signed and negative (non-ASCII byte)           */
                  [](unsigned char c)->char {
                      return static_cast<char>(std::tolower(c));
                  }
                 );

        /* Convert the word into a vector of strings (for now just individual
         * characters, then these will be iteratively replaced with strings of
         * multiple characters)                                                 */
        const auto word_size = word.size();
        vector<string> word_strvec(word_size);

        for (auto i = decltype(word_size){0}; i < word_size; ++i) {
            word_strvec.at(i) = word.at(i);
        }

        // Append the end-of-word character to the last character in the word
        word_strvec.at(word_size - 1) += (this->eow);

        /* Insert the word into the word list
         * NOTE: insert repeated words as well, since we are encoding a text,
         *       not building a vocabulary                                      */
        words_list.emplace_back(word_strvec);

        ++words_it;
    }


    /* -------------------------------------------------------------------------
     * 2. For each word in the text, loop over all symbol pairs (initially
     *    symbol=character, then characters will be iteratively merged into
     *    subwords) in the word. For each word:
     *      a. If at least one pair is part of the token-to-ID vocabulary, then
     *         choose the pair with lowest ID (i.e., the pair that was added to
     *         the token-to-ID vocabulary first) and replace the leftmost
     *         occurrence of that pair in the word with the merged symbol.
     *      b. Repeat step (a) until no pairs are found in the token-to-ID
     *         vocabulary.
     * -------------------------------------------------------------------------*/
    const auto nwords = words_list.size();
    
    if (nwords < 1) {
        throw runtime_error("bpe_tokenizer_t::encode(): need at least one word in the text");
        return vector<size_t>();  // Not reached
    }

    for (auto &word_strvec : words_list) {
        bool symbol_merges_possible = true;

        while (symbol_merges_possible) {
            auto   i_symbolpair_replace = word_strvec.size();  // Invalid (too large) index within the word
            string   symbolpair_replace = "";

            for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size() - 1; ++i) {
                const string symbol_pair = word_strvec.at(i) + word_strvec.at(i+1);
                try {
                    const auto id_symbol_pair = (this->vocab_token2id).at(symbol_pair);  // Throws an exception if 'symbol_pair' is not found in the token-to-ID vocabulary
                    i_symbolpair_replace = i;
                      symbolpair_replace = symbol_pair;
                } catch (const exception &e) {
                    #if (VERBOSE)
                    cout << "Skipping unknown symbol pair '" << symbol_pair
                         << "' while encoding the text (exception: \"" << e.what() << "\")" << endl;
                    #endif
                }
            }
    
            if (i_symbolpair_replace < word_strvec.size()) {
                assert(i_symbolpair_replace < word_strvec.size() - 1);  // Actually
                word_strvec.at(i_symbolpair_replace) = symbolpair_replace;
                word_strvec.erase(word_strvec.begin() + i_symbolpair_replace + 1);
            } else {
                symbol_merges_possible = false;
                #if (VERBOSE)
                cout << "No tokens found for this word in the token-to-ID vocabulary (this may just mean all tokens from the vocabulary have been merged into compound tokens)" << endl;
                #endif
            }
        }
    }


    /* --------------------
     * 3. Tokenize the text
     * -------------------- */
    // Get the number of tokens in the input text
    size_t ntokens = 0;
    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
            // Sanity check
            if (token == (this->eow)) {
                throw runtime_error("bpe_tokenizer_t::encode(): found end-of-word character token in the input text. This is not supported: please change the end-of-word character to something not present in the training text.");
                return vector<size_t>();  // Not reached
            }
            ++ntokens;
        }
    }

    vector<size_t> tokenIDs(ntokens);
    size_t m = 0;

    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
            /* Convert the token into the ID if found in the vocabulary,
             * otherwise set the ID to 'unknown'                                */
            try {
                /* NOTE: the at() method on the RHS throws an exception if the
                 *       token is not in the token-to-ID vocabulary             */
                tokenIDs.at(m) = (this->vocab_token2id).at(token);
            } catch (const exception &e) {
                tokenIDs.at(m) = (this->unk).second;
                cerr << "Unknown token '" << token << "': setting ID to 'unknown' token ID "
                     << (this->unk).second << " (exception: \"" << e.what() << "\")" << endl;
            }

            ++m;
        }
    }

    return tokenIDs;
}



/* ========================================================================
 * Decode method using the ID-to-token vocabulary to convert a set of input
 * token IDs into the corresponding text tokens
 * ========================================================================= */
string bpe_tokenizer_t::decode(const vector<size_t> &ids) {
    ostringstream decoded_text_ss;

    for (const auto &id : ids) {
        try {
            // NOTE: no extra space separating tokens
            decoded_text_ss << (this->vocab_id2token).at(id);
        } catch (const exception &e) {
            ostringstream exception_ss;
            exception_ss << "bpe_tokenizer_t::decode(): unknown token ID " << id
                         << ": this should never happen because the 'unknown' token should be part of the dictionary. Please check the code's correctness (exception: \""
                         << e.what() << "\")";
            throw runtime_error(exception_ss.str());
            return string();  // Not reached
        }
    }

    return decoded_text_ss.str();
}
