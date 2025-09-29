#include <vector>
#include <string>
#include <unordered_map>
#include <regex>
#include <algorithm>

#include "Vocabulary.hh"

using namespace std;


/* ============================================================================
 * Vocabulary built from an input text and consisting of a set of unique tokens
 * and a hashmap relating the tokens with unique IDs
 * ============================================================================ */
vocabulary_t::vocabulary_t(const string &text) {
    // This regex matches words, numbers, and punctuation as individual tokens
    regex                 token_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator it(text.begin(), text.end(), token_re);
    sregex_token_iterator end;

    size_t tokenID = 0;

    while (it != end) {
        /* Make all tokens lowercase to avoid duplicating them if the same
         * word appears multiple times with different cases                 */
        auto token = it->str();
        transform(token.begin(), token.end(), token.begin(), //::tolower);
                  /* Wrap ::tolower in a lambda to cast the input into
                   * unsigned char to avoid undefined behavior if the input
                   * char is signed and negative (non-ASCII byte)           */
                  [](unsigned char c) -> char {
                      return static_cast<char>(std::tolower(c));
                  }
                 );

        /* Try inserting the token into the vocabulary; this will only succeed
         * if the token is not in the vocabulary yet, since the tokens are the
         * keys in the map and keys are unique                                  */
        if (this->tokens_hashmap.emplace(token, tokenID).second) {
            this->tokens.push_back(token);
        }

        ++tokenID;
        ++it;
    }
}
