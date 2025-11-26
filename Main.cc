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
        ostringstream training_text_ss;
        training_text_ss << infile.rdbuf();
        const auto   &training_text(training_text_ss.str());
        const string &input_text(INPUT_TEXT);

        cout << "===============" << endl
             << "Basic tokenizer" << endl
             << "===============" << endl;

        auto tokenizer = tokenizer_t(training_text);

        cout << endl
             << "---------------------------------------" << endl
             << "Basic tokenizer: token-to-ID vocabulary" << endl
             << "---------------------------------------" << endl;

        for (const auto &[token, id] : tokenizer.vocab_token2id) {
            cout << token << "\t" << id << endl;
        }

        cout << endl
             << "---------------------------------------" << endl
             << "Basic tokenizer: ID-to-token vocabulary" << endl
             << "---------------------------------------" << endl;

        for (const auto &[id, token] : tokenizer.vocab_id2token) {
            cout << id << "\t" << token << endl;
        }

        cout << endl
             << "-------------------------------------------" << endl
             << "Basic tokenizer: text encoding and decoding" << endl
             << "-------------------------------------------" << endl;

        const auto &tokens = tokenizer.encode(input_text);

        cout << "***** Encoded text *****" << endl;
        for (const auto &token : tokens) {
            cout << token << " ";
        }

        cout << endl << "***** Decoded text *****" << endl;
        const auto &input_text_decoded = tokenizer.decode(tokens);
        cout << input_text_decoded << endl;


        cout << endl
             << "=============" << endl
             << "BPE tokenizer" << endl
             << "=============" << endl;

        auto tokenizer_bpe = tokenizer_bpe_t(training_text, BPE_END_OF_WORD, BPE_MAX_VOCAB_SIZE);

        cout << endl
             << "-------------------------------------" << endl
             << "BPE tokenizer: token-to-ID vocabulary" << endl
             << "-------------------------------------" << endl;

        for (const auto &[token, id] : tokenizer_bpe.vocab_token2id) {
            cout << token << "\t" << id << endl;
        }

        cout << endl
             << "-------------------------------------" << endl
             << "BPE tokenizer: ID-to-token vocabulary" << endl
             << "-------------------------------------" << endl;

        for (const auto &[id, token] : tokenizer_bpe.vocab_id2token) {
            cout << id << "\t" << token << endl;
        }

        cout << endl
             << "-------------------------------------------" << endl
             << "Basic tokenizer: text encoding and decoding" << endl
             << "-------------------------------------------" << endl;

        const auto &tokens_bpe = tokenizer_bpe.encode(input_text, BPE_END_OF_WORD);

        cout << "***** Encoded text *****" << endl;
        for (const auto &token : tokens_bpe) {
            cout << token << " ";
        }

        cout << endl << "***** Decoded text *****" << endl;
        const auto &input_text_decoded_bpe = tokenizer_bpe.decode(tokens_bpe);
        cout << input_text_decoded_bpe << endl;
    }

    // Handle file opening issues
    else {
        cerr << "Error opening file '" << INFILE_TRAINING << "'" << endl;
        return 1;
    }

    return 0;
}
