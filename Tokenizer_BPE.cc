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


/* ===========================================================================
 * Method to get the list of words split into subword tokens out of a text via
 * the byte-pair encoding (BPE) technique
 * ===========================================================================  */
void tokenizer_bpe_t::text2tokens(const string &text,
                                  const size_t &nmerges,
                                  const bool   &skip_repeated_words,
                                        vector<vector<string>> &words_list) {
    /* --------------------------------------------------------------
     * 1. Build a list of all the words in the input text as lists of
     *    individual characters
     * --------------------------------------------------------------           */
    /* This regex matches words, numbers, and punctuation as individual tokens,
     * which should be regarded as lists of "symbols"                           */
    regex                 word_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator words_it(text.begin(), text.end(), word_re);
    sregex_token_iterator end;

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
         * multiple characters)
         * NOTE: don't isolate the end-of-word character                        */
        const auto word_size = word.size();
        vector<string> word_strvec(word_size);
        for (auto i = decltype(word_size){0}; i < word_size; ++i) {
            word_strvec.at(i) = word.at(i);
        }
        word_strvec.at(word_size - 1) += END_OF_WORD_BPE;

        /* Insert the word into the word list (if 'skip_repeated_words' is true,
         * only do that if the word is not yet in the list)                     */
        if (skip_repeated_words) {
            if (find(words_list.begin(), words_list.end(), word_strvec) == words_list.end()) {
                words_list.emplace_back(word_strvec);
            }
        } else {
            words_list.emplace_back(word_strvec);
        }

        ++words_it;
    }

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Initial list of words in the text *****" << endl;
    //for (const auto &word_strvec : words_list) {
    //    for (const auto &el : word_strvec) {
    //        cout << el << " ";
    //    }
    //    cout << endl;
    //}
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX


    /* -------------------------------------------------------------------------
     * 2. For each word in the text, loop over all symbol pairs (initially
     *    symbol=character, then characters will be iteratively merged into
     *    subwords) in the word and build a temporary map of
     *    symbol-pairs->frequency. Then, find the symbol pair that occurs the
     *    most and replace its separate symbols in the words list with that
     *    symbol pair (a single string). Repeat 'nmerges' times, where 'nmerges'
     *    is specified by the user, each time updating the words list.
     * -------------------------------------------------------------------------*/
    const auto nwords = words_list.size();

    if (nwords < 1) {
        throw runtime_error("Need at least one word in the text");
    }

    for (auto n = decltype(nmerges){0}; n < nmerges; ++n) {
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
                        exception_ss << "Failed to access the symbol-pair->frequency map at key '" << symbol_pair <<
                            "'. This key should exist, please check the code (exception: \"" << e.what() << "\")." << endl;
                        throw runtime_error(exception_ss.str());
                    }
                }
            }
        }

        const auto iterator_maxcounts = max_element(symbolpairs_freq.begin(), symbolpairs_freq.end(),
                                                    [](const auto &a, const auto &b){
                                                        return (a.second < b.second);
                                                    });
        const auto &most_common_symbolpair = iterator_maxcounts->first;

        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX: debug only
        //cout << "----------" << endl
        //     << "n = " << n  << endl
        //     << "----------" << endl
        //     << "Most common symbol pair: " << most_common_symbolpair << ", frequency " << iterator_maxcounts->second << endl
        //     << "Symbol-pair->frequency map:" << endl;
        //// ***** WARNING *****
        //// This may potentially print A LOT of text
        //// *******************
        ////for (const auto &[symbol_pair, freq] : symbolpairs_freq) {
        ////    cout << symbol_pair << ", " << freq << endl;
        ////}
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX


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

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Final list of words in the training text *****" << endl;
    //for (const auto &word_strvec : words_list) {
    //    for (const auto &el : word_strvec) {
    //        cout << el << " ";
    //    }
    //    cout << endl;
    //}
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX

    return;
}



/* ===========================================================================
 * Constructor building a map relating each subword token in the training text
 * to how many times that subword occurs in the traning text
 * ===========================================================================  */
tokenizer_bpe_t::tokenizer_bpe_t(const string &training_text,
                                 const size_t &nmerges_training) {
    // Get a list of words split into subwords tokens from the training text
    vector<vector<string>> words_list;
    const bool skip_repeated_words = true;  // Don't want duplicate words in the vocabulary
    this->text2tokens(training_text, nmerges_training, skip_repeated_words, words_list);

    /* -------------------------------------------------------------------
     * Loop over the words list and link each token (=subword) to a unique
     * integer ID through a hash map (as well as its inverse map)
     * -------------------------------------------------------------------     */
    size_t id = 0;

    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
            // Sanity check
            if (token == END_OF_WORD_BPE) {
                throw runtime_error("Found end-of-word character token in the training text. This should never happen: please check the code does not isolate the end-of-word character when splitting each word into individual characters during training.");
            }

            /* Try inserting the token into the token-to-ID vocabulary (O(1) for
             * token lookup in tokenizer_BPE_t::encode()); this will only
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

    /* Add extra IDs to handle unknown and end-of-text (useful when training
     * with multiple text sources) tokens (set frequencies to 1)                */
    const auto original_vocab_size = (this->vocab_token2id).size();
    (this->unk) = {"<|unknown|>",     original_vocab_size};
    (this->eot) = {"<|end-of-text|>", original_vocab_size + 1};

    if (not (this->vocab_token2id).emplace(this->unk).second) {
        throw runtime_error("Insertion of 'unknown' token failed");
    }

    if (not (this->vocab_token2id).emplace(this->eot).second) {
        throw runtime_error("Insertion of 'end-of-text' token failed");
    }

    /* Build the ID-to-subword vocabulary for fast (O(1)) ID lookup in
     * tokenizer_BPE_t::decode()                                                */
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
vector<size_t> tokenizer_bpe_t::encode(const string &text,
                                       const size_t &nmerges) {
    /* Get a list of words split into subwords tokens from the input text
     * NOTE: a number of merges different from that used during training is
     *       allowed (may or may not be beneficial, try it and see)             */
    vector<vector<string>> words_list;
    const bool skip_repeated_words = false;  // Encode duplicate words too
    this->text2tokens(text, nmerges, skip_repeated_words, words_list);

    // Get the number of tokens in the input text
    size_t ntokens = 0;
    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
            // Sanity check
            if (token == END_OF_WORD_BPE) {
                throw runtime_error("Found end-of-word character token in the input text. This is not supported: either change the end-of-word character to something not present in the input text, or remove the end-of-word character from the input text.");
            }
            ++ntokens;
        }
    }

    vector<size_t> tokenIDs(ntokens);

    size_t i = 0;
    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
            /* Convert the token into the ID if found in the vocabulary,
             * otherwise set the ID to 'unknown'                                */
            try {
                /* NOTE: the at() method on the RHS throws an exception if the
                 *       token is not in the token-to-ID vocabulary             */
                tokenIDs.at(i) = (this->vocab_token2id).at(token);
            } catch (exception &e) {
                tokenIDs.at(i) = (this->unk).second;
                cerr << "Unknown token '" << token << "': setting ID to 'unknown' token ID "
                     << (this->unk).second << " (exception: \"" << e.what() << "\")" << endl;
            }

            ++i;
        }
    }

    return tokenIDs;
}



/* ========================================================================
 * Decode method using the ID-to-token vocabulary to convert a set of input
 * token IDs into the corresponding text tokens
 * ========================================================================= */
string tokenizer_bpe_t::decode(const vector<size_t> &ids) {
    ostringstream decoded_text_ss;

    for (const auto &id : ids) {
        try {
            /* NOTE: extra space separating tokens just to clarify which tokens
             *       are being used to build the decoded text                   */
            decoded_text_ss << (this->vocab_id2token).at(id) << " ";
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
