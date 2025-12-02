#include <cassert>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

#include "Check_parameters.hh"
#include "Types.hh"
#include "Parameters.hh"

using namespace std;


int main() {
    ifstream infile(INFILE_TRAINING, ifstream::in);

    if (infile.is_open()) {
        // Encode the input text into token IDs
        ostringstream training_text_ss;
        training_text_ss << infile.rdbuf();
        const auto   &training_text(training_text_ss.str());
        const string &input_text(INPUT_TEXT);

        #if (TOKENIZER == WORD)
        auto tokenizer = word_tokenizer_t(training_text);
        #elif (TOKENIZER == BPE)
        auto tokenizer = bpe_tokenizer_t(training_text, BPE_END_OF_WORD, BPE_MAX_VOCAB_SIZE);
        #else
        #error "Invalid tokenizer"
        return 1;  // Not reached
        #endif

        const auto &ids = tokenizer.encode(input_text);
        const auto nids = ids.size();


        /* Initialize the input matrix and the query, key, and value weight
         * matrices to random numbers in [0,1]                                  */
        const     auto dim_inputs = nids*DIM_IN;
        constexpr auto dim_inout  = DIM_IN*DIM_OUT;

        vector<double> inputs(dim_inputs);
        array<double, dim_inout> Wq, Wk, Wv;

        random_device rd;  // Use machine entropy as the random seed
        // XXX
        //const auto seed = rd();
        const auto seed = 1.;
        // XXX
        mt19937 gen(seed);
        uniform_real_distribution<double> dist(0., 1.);

        for (auto m = decltype(nids){0}; m < nids; ++m) {
            const auto idx_m = m*DIM_IN;
            for (auto i = decltype(DIM_IN){0}; i < DIM_IN; ++i) {
                 inputs.at(idx_m + i) = dist(gen);
            }
        }

        for (auto i = decltype(DIM_IN){0}; i < DIM_IN; ++i) {
            const auto idx_i = i*DIM_OUT;
            for (auto j = decltype(DIM_OUT){0}; j < DIM_OUT; ++j) {
                const auto ij = idx_i + j;
                Wq.at(ij) = dist(gen);
                Wk.at(ij) = dist(gen);
                Wv.at(ij) = dist(gen);
            }
        }


        // Build the query, key, and value matrices
        const auto dim_qkv = nids*DIM_OUT;
        vector<double> queries(dim_qkv, 0.), keys(dim_qkv, 0.), values(dim_qkv, 0.);  // NOTE: initialize to zero

        for (auto m = decltype(nids){0}; m < nids; ++m) {
            const auto idx_m_in  = m*DIM_IN;
            const auto idx_m_out = m*DIM_OUT;

            /* NOTE: swapping the more "natural" loop order (j out, k in) to
             *       improve the memory access pattern in Wq, Wk, Wv            */
            for (auto k = decltype(DIM_IN){0}; k < DIM_IN; ++k) {
                const auto inputs_mk = inputs.at(idx_m_in + k);
                const auto idx_k     = k*DIM_OUT;

                for (auto j = decltype(DIM_OUT){0}; j < DIM_OUT; ++j) {
                    const auto mj = idx_m_out + j;
                    const auto kj = idx_k + j;

                    queries.at(mj) += inputs_mk*Wq.at(kj);
                       keys.at(mj) += inputs_mk*Wk.at(kj);
                     values.at(mj) += inputs_mk*Wv.at(kj);
                }
            }
        }


        /* Compute the attention scores and the context vectors (matrix), i.e.,
         * the sum of the value vectors (columns of the values matrix) weighted
         * by the attention scores along the rows of the attention matrix
         * NOTE: no need to store the full attention matrix: only compute each
         *       row and the corresponding context vector                       */
        constexpr auto sqrt_dim_out_inv = 1./sqrt(static_cast<double>(DIM_OUT));
        vector<double> contexts(dim_qkv);  // nids*DIM_OUT

        for (auto m = decltype(nids){0}; m < nids; ++m) {
            const auto idx_m_out = m*DIM_OUT;

            /* Causal attention: each token ID in the input text only attends to
             * all the previous ones, so that the attention scores in the upper
             * triangular part of the attention scores matrix (i.e., all the
             * attention scores for n > m for row/token m) are zero (not even
             * defined here)                                                    */
            vector<double> attention_m(m+1);  // Instead of attention_m(nids)

            for (auto n = decltype(nids){0}; n <= m; ++n) {
                const auto idx_n_out = n*DIM_OUT;
                double attention_mn  = 0.;

                for (auto l = decltype(DIM_OUT){0}; l < DIM_OUT; ++l) {
                    const auto ml = idx_m_out + l;
                    const auto nl = idx_n_out + l;
                    attention_mn += queries.at(ml)*keys.at(nl);
                }

                /* Scale the attention score by
                 * sqrt(len(keys[:,1]) = sqrt(DIM_OUT) to improve training
                 * behavior later on                                        */
                attention_m.at(n) = attention_mn*sqrt_dim_out_inv;
            }


            /* Normalize the attention scores for the current token (i.e., for
             * the current row of the attention matrix) using a stabilized
             * softmax                                                          */
            auto attention_m_max = -numeric_limits<double>::infinity();

            for (auto n = decltype(nids){0}; n <= m; ++n) {
                if (attention_m.at(n) > attention_m_max) {
                    attention_m_max = attention_m.at(n);
                }
            }

            assert(isfinite(attention_m_max));
            double sum_exp = 0.;

            for (auto n = decltype(nids){0}; n <= m; ++n) {
                double exp_att    = exp(attention_m.at(n) - attention_m_max);
                attention_m.at(n) = exp_att;
                sum_exp          += exp_att;
            }

            assert(sum_exp > 0.);


            /* Finish normalizing the attention scores for the current token (i.e.,
             * for the current row of the attention matrix) and calculate the
             * context vector for the current token
             * NOTE: swapping the more "natural" loop order (j out, n in) to
             *       improve the memory access pattern in the values matrix     */
            for (auto n = decltype(nids){0}; n <= m; ++n) {
                const auto idx_n_out    = n*DIM_OUT;
                const auto attention_mn = attention_m.at(n)/sum_exp;  // NOTE: attention_m.at(n) is NOT normalized

                for (auto j = decltype(DIM_OUT){0}; j < DIM_OUT; ++j) {
                    contexts.at(idx_m_out + j) += attention_mn*values.at(idx_n_out + j);
                }
            }
        }

        // XXX
        for (auto m = decltype(nids){0}; m < nids; ++m) {
            const auto idx_m_out = m*DIM_OUT;
            for (auto j = decltype(DIM_OUT){0}; j < DIM_OUT; ++j) {
                cout << contexts.at(idx_m_out + j) << "\t";
            } cout << endl;
        }
        // XXX








        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        //for (const auto &token : tokens) {
        //    cout << token << " ";
        //} cout << endl;

        //cout << endl << "***** Decoded text *****" << endl;
        //const auto &input_text_decoded = tokenizer.decode(tokens);
        //cout << input_text_decoded << endl;
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX


        // TODO: generate input-output token ID pairs (input must have size CONTEXT_SIZE)
        //if (CONTEXT_SIZE >= nids) {
        //    cerr << "Context size (" << CONTEXT_SIZE << ") must be smaller than the number of tokens in the training text (" << nids << ")" << endl;
        //    return 1;
        //} else {
        //    const auto n_io = nids - CONTEXT_SIZE + 1;
        //    assert(n_io > 0);

        //    vector<array<size_t, CONTEXT_SIZE + 1>> ios(n_io);

        //    for (auto n = decltype(n_io){0}; n < n_io; ++n) {
        //        auto &io = ios.at(n);
        //        output = tokens.at(n + CONTEXT_SIZE);

        //        for (auto i = decltype(CONTEXT_SIZE){0}; i < CONTEXT_SIZE; ++i) {
        //            io.at(i) = tokens.at(n + i);
        //        }

        //        io.at(CONTEXT_SIZE) = 
        //    }

        //    // XXX
        //    //for (const auto &[input, output] : ios) {
        //    //    for (const auto &token : input) {
        //    //        cout << token << " ";
        //    //    }
        //    //    cout << "\t" << output << endl;
        //    //}
        //    // XXX
        //}
    }

    // Handle file opening issues
    else {
        cerr << "Unable to open file '" << INFILE_TRAINING << "'" << endl;
        return 1;
    }

    return 0;
}
