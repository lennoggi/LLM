#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <regex>

#include "Types.hh"
#include "Parameters.hh"

using namespace std;


/* =============================================================================
 * Constructor building a map relating each word (initially a list of individual
 * characters which are iteratively merged into subwords) to how many times that
 * word occurs in the traning text
 * =============================================================================*/
tokenizer_bpe_t::tokenizer_bpe_t(const string &training_text,
                                 const size_t &nmerges) {
    /* ------------------------------------------------------------------------
     * 1. Build a map relating each word (as a list of individual characters
     *    plus the end-of-word character) to how many times that word occurs in
     *    the traning text
     * ------------------------------------------------------------------------  */
    /* This regex matches words, numbers, and punctuation as individual tokens,
     * which should be regarded as lists of "symbols"                           */
    regex                 token_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator tokens_it(training_text.begin(), training_text.end(), token_re);
    sregex_token_iterator end;

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

        /* Add the 'end-of-word' character (ASCII separator) to the word
         * NOTE: usually not printable with std::cout (print either produces
         *       nothing or weird strings)                                      */
        token += 0x1F;

        /* Convert the string token into a vector of strings (for now just
         * individual characters, then these will be iteratively replaced with
         * strings)                                                             */
        vector<string> token_strvec(token.size());
        for (auto i = decltype(token.size()){0}; i < token.size(); ++i) {
            token_strvec.at(i) = token.at(i);
        }

        /* Try inserting the word into the symbols->occurences map (where
         * initially "symbols" are just words regarded as lists of symbols) with
         * an initial number of occurences of 1; this will only succeed if the
         * word is not in the map yet, since the words are the keys in the map
         * and keys are unique. If insertion fails, add 1 to the number of
         * occurences of the given word.                                        */
        if (not (this->symbols_occurences).emplace(token_strvec, 1).second) {
            try {
                (this->symbols_occurences).at(token_strvec) += 1;
            } catch (const exception &e) {
                ostringstream exception_ss;
                exception_ss << "Failed to access the symbols->occurences map at key '" << token <<
                    "'. This key should exist, please check the code (exception: \"" << e.what() << "\")." << endl;
                throw runtime_error(exception_ss.str());
            }
        }

        ++tokens_it;
    }

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Initial vocabulary *****" << endl;
    //for (const auto &[token_strvec, occurrences] : (this->symbols_occurences)) {
    //    for (const auto &el : token_strvec) {
    //        cout << el << " ";
    //    }
    //    cout << ", freq: " << occurrences << endl;
    //}
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX


    /* -------------------------------------------------------------------------
     * 2. For each word (i.e., list of symbols; initially symbol=character, may
     *    become a subword later iteratively) in the symbols->occurrences map,
     *    loop over all symbols pairs in the word and build a separate map of
     *    symbol pairs->occurrences. Then, find the symbol pair that occurs the
     *    most and replace its separate symbols in the original map (the
     *    "symbols->occurrences" map) with the symbol pair (a single string).
     *    Repeat 'nmerges' times (where 'nmerges' is specified by the user),
     *    each time updating the symbols->occurrences map.
     * -------------------------------------------------------------------------*/
    const auto ntokens = (this->symbols_occurences).size();

    if (ntokens < 1) {
        throw runtime_error("Need at least one token in the training text");
    }

    for (auto n = decltype(nmerges){0}; n < nmerges; ++n) {
        unordered_map<string, size_t> symbolpairs_occurences;

        for (const auto &[token_strvec, occurrences] : (this->symbols_occurences)) {
            for (auto i = decltype(token_strvec.size()){0}; i < token_strvec.size()-1; ++i) {
                const string symbol_pair = token_strvec.at(i) + token_strvec.at(i+1);

                /* Try adding 'symbol_pair' to 'symbolpairs_occurences' with
                 * an initial occurence of 1; if that fails, that means the
                 * symbol pair is already in the map, so increment its
                 * occurences by 1.                                             */
                if (not symbolpairs_occurences.emplace(symbol_pair, 1).second) {
                    try {
                        symbolpairs_occurences.at(symbol_pair) += 1;
                    } catch (const exception &e) {
                        ostringstream exception_ss;
                        exception_ss << "Failed to access the symbol_pair->occurences map at key '" << symbol_pair <<
                            "'. This key should exist, please check the code (exception: \"" << e.what() << "\")." << endl;
                        throw runtime_error(exception_ss.str());
                    }
                }
            }
        }

        const auto iterator_maxcounts = max_element(symbolpairs_occurences.begin(), symbolpairs_occurences.end(),
                                                    [](const auto &a, const auto &b){
                                                        return (a.second < b.second);
                                                    });
        const auto &most_common_symbolpair = iterator_maxcounts->first;

        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX: debug only
        //cout << "n=" << n << ", most common symbol pair: (" << most_common_symbolpair << ", " << iterator_maxcounts->second << ")" << endl;
        //cout << "Symbol pair -> Occurences map for n=" << n << ":" << endl;
        //for (const auto &[symbol_pair, occurrences] : symbolpairs_occurences) {
        //    cout << symbol_pair << ", " << occurrences << endl;
        //}
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX


        // Update the symbols->occurences map for the next iteration
        for (auto &[token_strvec, occurrences] : (this->symbols_occurences)) {
            for (auto i = decltype(token_strvec.size()){0}; i < token_strvec.size()-1; ++i) {
                const string symbol_pair = token_strvec.at(i) + token_strvec.at(i+1);
                if (symbol_pair == most_common_symbolpair) {
                    // NOTE: this assignements would be illegal because the map's key it's implicitly const
                    //token_strvec.at(i) = most_common_symbolpair;
                    auto node_handler = (this->symbols_occurences).extract(token_strvec);
                    node_handler.key().at(i) = most_common_symbolpair;
                    /* Remove the second character of the pair from the "word"
                     * (list of symbols) so it can be deleted (see below)       */
                    //token_strvec.at(i+1) = "";  // Illegal (see above)
                    node_handler.key().at(i+1) = "";
                    (this->symbols_occurences).insert(move(node_handler));
                }
            }

            /* Delete empty strings (empty because they have been merged above)
             * from the "word" (list of symbols)                                */
            for (auto i = decltype(token_strvec.size()){0}; i < token_strvec.size(); ++i) {
                if (token_strvec.at(i) == "") {
                    auto node_handler = (this->symbols_occurences).extract(token_strvec);
                    //token_strvec.erase(token_strvec.begin() + i);  // Illegal (see above)
                    node_handler.key().erase(token_strvec.begin() + i);
                    (this->symbols_occurences).insert(move(node_handler));
                }
            }
        }
    }

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Final vocabulary *****" << endl;
    //for (const auto &[token_strvec, occurrences] : (this->symbols_occurences)) {
    //    for (const auto &el : token_strvec) {
    //        cout << el << " ";
    //    }
    //    cout << ", freq: " << occurrences << endl;
    //}
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
}



// TODO: implement the encode and decode methods
