#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Vocabulary.hh"
#include "Parameters.hh"

using namespace std;


int main() {
    ifstream infile(INFILE, ifstream::in);

    if (infile.is_open()) {
        ostringstream oss;
        oss << infile.rdbuf();
        const auto &voc = vocabulary_t(oss.str());

        for (const auto &token : voc.tokens) {
            cout << token << endl;
        }

        for (const auto &[token, tokenID] : voc.tokens_hashmap) {
            cout << token << "\t" << tokenID << endl;
        }

        const auto ntokens = voc.tokens.size();
        cout << "There are " << ntokens << " unique tokens in the input text" << endl;
        assert(voc.tokens_hashmap.size() == ntokens);
    }

    else {
        cerr << "Error opening file '" << INFILE << "'" << endl;
        return 1;
    }

    return 0;
}
