#include <vector>
#include <string>
#include <regex>
#include <algorithm>

#include "Declarations.hh"

using namespace std;


/* =====================================================
 * Encode method to convert a text into unique token IDs
 * ===================================================== */
vector<string> encode(const string &text) {
    // This regex matches words, numbers, and punctuation as individual tokens
    regex                 token_re(R"([a-zA-Z0-9]+|[.,;:!?'"()\[\]\{\}\/\\])");
    sregex_token_iterator it(text.begin(), text.end(), token_re);
    sregex_token_iterator end;
    vector<string>        tokens;

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
        tokens.push_back(token);
        ++it;
    }

    // Sort the tokens and erase duplicates
    sort(tokens.begin(), tokens.end());
    tokens.erase(unique(tokens.begin(), tokens.end()),
                 tokens.end());

    return tokens;
}
