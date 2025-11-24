#include <iostream>
#include <fstream>
#include <sstream>

#include "Types.hh"
#include "Parameters.hh"

using namespace std;


int main() {
    ifstream infile(INFILE_TRAINING, ifstream::in);

    if (infile.is_open()) {
        /* Encode the input training text into a token-to-ID and an ID-to-token
         * vocabularies and print the vocabularies                              */
        ostringstream training_text_ss;
        training_text_ss << infile.rdbuf();
        const auto &training_text = training_text_ss.str();
        auto tokenizer = tokenizer_t(training_text);

        cout << "=======================================" << endl
             << "Basic tokenizer: token-to-ID vocabulary" << endl
             << "=======================================" << endl;

        for (const auto &[token, id] : tokenizer.vocab_token2id) {
            cout << token << "\t" << id << endl;
        }

        cout << endl
             << "=======================================" << endl
             << "Basic tokenizer: ID-to-token vocabulary" << endl
             << "=======================================" << endl;

        for (const auto &[id, token] : tokenizer.vocab_id2token) {
            cout << id << "\t" << token << endl;
        }

        cout << endl << "---------------------------------------------------------------------------------" << endl << endl;


        // Test the encode and decode methods onto a user-specified input text
        string input_text(INPUT_TEXT);
        const auto &tokens = tokenizer.encode(input_text);

        for (const auto &token : tokens) {
            cout << token << " ";
        }

        const auto &input_text_decoded = tokenizer.decode(tokens);
        cout << endl << input_text_decoded << endl;


        // Test the Bype-pair encoding (BPE) tokenizer
        auto tokenizer_bpe = tokenizer_bpe_t(training_text, NMERGES);

        cout << endl
             << "======================================================================" << endl
             << "BPE tokenizer: token pairs -> Occurences after " << NMERGES << " character merges" << endl
             << "======================================================================" << endl;

        for (const auto &[symbols, occurrences] : tokenizer_bpe.symbols_occurences) {
            for (const auto &symbol : symbols) {
                cout << symbol << " ";
            }
            cout << "\t" << occurrences << endl;
        }
    }

    // Handle file opening issues
    else {
        cerr << "Error opening file '" << INFILE_TRAINING << "'" << endl;
        return 1;
    }

    return 0;
}
