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
 * Constructor building a map relating each subword token in the training text
 * to how many times that subword occurs in the traning text
 * ===========================================================================  */
tokenizer_bpe_t::tokenizer_bpe_t(const string &training_text,
                                 const size_t &nmerges) {
    /* ----------------------------------------------------------------
     * 1. Build a list of all the words in the training set as lists of
     *    individual characters
     * ----------------------------------------------------------------         */
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
        transform(word.begin(), word.end(), word.begin(), //::tolower);
                  /* Wrap ::tolower in a lambda to cast the input into
                   * unsigned char to avoid undefined behavior if the input
                   * char is signed and negative (non-ASCII byte)           */
                  [](unsigned char c)->char {
                      return static_cast<char>(std::tolower(c));
                  }
                 );

        /* Add the 'end-of-word' character (ASCII separator) to the word
         * NOTE: usually not printable with std::cout (print either produces
         *       nothing or weird strings)                                      */
        word += 0x1F;

        /* Convert the word into a vector of strings (for now just individual
         * characters, then these will be iteratively replaced with strings of
         * multiple characters)                                                 */
        vector<string> word_strvec(word.size());
        for (auto i = decltype(word.size()){0}; i < word.size(); ++i) {
            word_strvec.at(i) = word.at(i);
        }

        // Insert the word into the word list if not there yet
        if (find(words_list.begin(), words_list.end(), word_strvec) == words_list.end()) {
            words_list.emplace_back(word_strvec);
        }

        ++words_it;
    }

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Initial list of words in the training text *****" << endl;
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
     * 2. For each word in the training text, loop over all symbol pairs
     *    (initially symbol=character, then characters will be iteratively
     *    merged into subwords) in the word and build a temporary map of
     *    symbol-pairs->frequency. Then, find the symbol pair that occurs the
     *    most and replace its separate symbols in the words list with that
     *    symbol pair (a single string). Repeat 'nmerges' times, where 'nmerges'
     *    is specified by the user, each time updating the words list.
     * -------------------------------------------------------------------------*/
    const auto nwords = words_list.size();

    if (nwords < 1) {
        throw runtime_error("Need at least one word in the training text");
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


    /* ------------------------------------------------------------------------
     * 3. Loop over the words list and link each token (=subword) to a unique
     *    integer ID through a hash map (as well as its inverse map)
     * ------------------------------------------------------------------------ */
    size_t id = 0;

    for (const auto &word_strvec : words_list) {
        for (const auto &token : word_strvec) {
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



// TODO: implement the encode and decode methods
