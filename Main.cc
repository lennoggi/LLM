#include <iostream>
#include <fstream>
#include <sstream>

#include "Check_parameters.hh"
#include "Types.hh"
#include "Parameters.hh"

using namespace std;


int main() {
    ifstream infile(INFILE_TRAINING, ifstream::in);

    if (infile.is_open()) {
        /* ---------------------
         * Simple tokenizer test
         * --------------------- */
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



        /* ---------------------------------------
         * Byte-pair encoding (BPE) tokenizer test
         * --------------------------------------- */
        auto tokenizer_bpe = tokenizer_bpe_t(training_text, NMERGES_BPE);

        cout << endl
             << "==========================" << endl
             << "BPE tokenizer: token-to-ID" << endl
             << "==========================" << endl;

        for (const auto &[token, id] : tokenizer_bpe.vocab_token2id) {
            cout << token << "\t" << id << endl;
        }

        cout << endl
             << "=====================================" << endl
             << "BPE tokenizer: ID-to-token vocabulary" << endl
             << "=====================================" << endl;

        for (const auto &[id, token] : tokenizer_bpe.vocab_id2token) {
            cout << id << "\t" << token << endl;
        }

        cout << endl << "---------------------------------------------------------------------------------" << endl << endl;

        // Test the encode and decode methods onto a user-specified input text
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        const auto &tokens_bpe = tokenizer_bpe.encode(input_text, NMERGES_BPE);
        //const auto &tokens_bpe = tokenizer_bpe.encode(input_text, 5);
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX

        for (const auto &token : tokens_bpe) {
            cout << token << " ";
        }

        const auto &input_text_decoded_bpe = tokenizer_bpe.decode(tokens_bpe);
        cout << endl << input_text_decoded_bpe << endl;
    }

    // Handle file opening issues
    else {
        cerr << "Error opening file '" << INFILE_TRAINING << "'" << endl;
        return 1;
    }

    return 0;
}
