#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <regex>
#include <algorithm>

#include "Types.hh"
#include "Parameters.hh"

using namespace std;


/* ============================================================================
 * Constructor building a map relating each subword in the training text to how
 * many times that subword occurs in the traning text
 * ============================================================================ */
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
        if (find((this->words_list).begin(), (this->words_list).end(), word_strvec) == (this->words_list).end()) {
            (this->words_list).emplace_back(word_strvec);
        }

        ++words_it;
    }

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Initial list of words in the training text *****" << endl;
    //for (const auto &word_strvec : (this->words_list)) {
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
     *    (initiallysymbol=character, then characters will be iteratively merged
     *    into subwords) in the word and build a temporary map of
     *    symbol-pairs->occurrences. Then, find the symbol pair that occurs the
     *    most and replace its separate symbols in the words list with that
     *    symbol pair (a single string). Repeat 'nmerges' times, where 'nmerges'
     *    is specified by the user, each time updating the words list.
     * -------------------------------------------------------------------------*/
    const auto nwords = (this->words_list).size();

    if (nwords < 1) {
        throw runtime_error("Need at least one word in the training text");
    }

    for (auto n = decltype(nmerges){0}; n < nmerges; ++n) {
        unordered_map<string, size_t> symbolpairs_occurences;

        for (const auto &word_strvec : (this->words_list)) {
            for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size()-1; ++i) {
                const string symbol_pair = word_strvec.at(i) + word_strvec.at(i+1);

                /* Try adding 'symbol_pair' to 'symbolpairs_occurences' with
                 * an initial occurence of 1; if that fails, that means the
                 * symbol pair is already in the map, so increment its
                 * occurences by 1.                                             */
                if (not symbolpairs_occurences.emplace(symbol_pair, 1).second) {
                    try {
                        symbolpairs_occurences.at(symbol_pair) += 1;
                    } catch (const exception &e) {
                        ostringstream exception_ss;
                        exception_ss << "Failed to access the symbol-pair->occurences map at key '" << symbol_pair <<
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
        //cout << "----------" << endl
        //     << "n = " << n  << endl
        //     << "----------" << endl
        //     << "Most common symbol pair: " << most_common_symbolpair << ", frequency " << iterator_maxcounts->second << endl
        //     << "Symbol-pair->occurences map:" << endl;
        //// ***** WARNING *****
        //// This may potentially print A LOT of text
        //// *******************
        ////for (const auto &[symbol_pair, occurrences] : symbolpairs_occurences) {
        ////    cout << symbol_pair << ", " << occurrences << endl;
        ////}
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX


        // Update the words list for the next iteration
        for (auto &word_strvec : (this->words_list)) {
            for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size()-1; ++i) {
                const string symbol_pair = word_strvec.at(i) + word_strvec.at(i+1);
                if (symbol_pair == most_common_symbolpair) {
                    word_strvec.at(i)   = most_common_symbolpair;
                    word_strvec.at(i+1) = "";
                    //// NOTE: this assignements would be illegal because the map's key it's implicitly const
                    ////word_strvec.at(i) = most_common_symbolpair;
                    //auto node_handler = (this->words_list).extract(word_strvec);
                    //node_handler.key().at(i) = most_common_symbolpair;
                    ///* Remove the second character of the pair from the "word"
                    // * (list of symbols) so it can be deleted (see below)       */
                    ////word_strvec.at(i+1) = "";  // Illegal (see above)
                    //node_handler.key().at(i+1) = "";
                    //(this->words_list).insert(move(node_handler));
                }
            }

            /* Delete empty strings (empty because they have been merged above)
             * from the "word" (list of symbols)                                */
            for (auto i = decltype(word_strvec.size()){0}; i < word_strvec.size(); ++i) {
                if (word_strvec.at(i) == "") {
                    word_strvec.erase(word_strvec.begin() + i);
                    //auto node_handler = (this->words_list).extract(word_strvec);
                    ////word_strvec.erase(word_strvec.begin() + i);  // Illegal (see above)
                    //node_handler.key().erase(word_strvec.begin() + i);
                    //(this->words_list).insert(move(node_handler));
                }
            }
        }
    }

    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: debug only
    //cout << "***** Final list of words in the training text *****" << endl;
    //for (const auto &word_strvec : (this->words_list)) {
    //    for (const auto &el : word_strvec) {
    //        cout << el << " ";
    //    }
    //    cout << endl;
    //}
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
}



// TODO: implement the encode and decode methods
