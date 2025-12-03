#include <cassert>
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
        /* Build the tokenizer object, which contains the token<->ID
         * vocabularies and the encode() and decode() methods                   */
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


        /* Initialize a vector representation ("embedding") of each token in the
         * vocabulary with random numbers (to be optimized during training
         * later on)                                                            */
        const auto nids_vocab = tokenizer.vocab_token2id.size();
        vector<double> vocab_embedding(nids_vocab*DIM_IN);

        random_device rd;
        mt19937 gen(rd());  // Use machine entropy as the random seed
        normal_distribution<double> ndist(0., 1./sqrt(static_cast<double>(DIM_IN)));

        for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
            const auto idx_v = v*DIM_IN;
            for (auto i = decltype(DIM_IN){0}; i < DIM_IN; ++i) {
                vocab_embedding.at(idx_v + i) = ndist(gen);
            }
        }


        /* Encode the input text into token IDs and map each ID into the
         * corresponding embedding vector, scaled by sqrt(DIM_IN) to keep
         * magnitudes consistent                                                */
        const auto &ids_input = tokenizer.encode(input_text);
        const auto nids_input = ids_input.size();
        const auto dim_inputs = nids_input*DIM_IN;

        vector<double> inputs(dim_inputs);
        const auto sqrt_dim_in = sqrt(static_cast<double>(DIM_IN));

        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            const auto idx_m_in  = m*DIM_IN;
            const auto id_input  = ids_input.at(m);
            assert(id_input < nids_vocab);
            const auto idx_vocab = id_input*DIM_IN;

            for (auto i = decltype(DIM_IN){0}; i < DIM_IN; ++i) {
                 inputs.at(idx_m_in + i) = sqrt_dim_in*vocab_embedding.at(idx_vocab + i);
            }
        }


        /* Initialize the query, key, and value weight matrices to random values
         * NOTE: declared as std::vector so they are allocated on the heap.
         *       std::array allocates on the stack and this can overflow for
         *       very large DIM_IN*DIM_OUT.                                     */
        constexpr auto dim_inout = DIM_IN*DIM_OUT;
        vector<double> Wq(dim_inout), Wk(dim_inout), Wv(dim_inout);

        // Xavier/Glorot distribution
        constexpr auto xg_bound = sqrt(6./(static_cast<double>(DIM_IN) + static_cast<double>(DIM_OUT)));
        uniform_real_distribution<double> xgdist(-xg_bound, xg_bound);

        for (auto i = decltype(DIM_IN){0}; i < DIM_IN; ++i) {
            const auto idx_i = i*DIM_OUT;
            for (auto j = decltype(DIM_OUT){0}; j < DIM_OUT; ++j) {
                const auto ij = idx_i + j;
                Wq.at(ij) = xgdist(gen);
                Wk.at(ij) = xgdist(gen);
                Wv.at(ij) = xgdist(gen);
            }
        }


        // Build the query, key, and value matrices
        const auto dim_qkv = nids_input*DIM_OUT;
        vector<double> queries(dim_qkv, 0.), keys(dim_qkv, 0.), values(dim_qkv, 0.);  // NOTE: initialize to zero

        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
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

        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            const auto idx_m_out = m*DIM_OUT;

            /* Causal attention: each token ID in the input text only attends to
             * all the previous ones, so that the attention scores in the upper
             * triangular part of the attention scores matrix (i.e., all the
             * attention scores for n > m for row/token m) are zero (not even
             * defined here)                                                    */
            vector<double> attention_m(m+1);  // Instead of attention_m(nids)

            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
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

            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                if (attention_m.at(n) > attention_m_max) {
                    attention_m_max = attention_m.at(n);
                }
            }

            assert(isfinite(attention_m_max));
            double sum_exp = 0.;

            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
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
            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                const auto idx_n_out    = n*DIM_OUT;
                const auto attention_mn = attention_m.at(n)/sum_exp;  // NOTE: attention_m.at(n) is NOT normalized

                for (auto j = decltype(DIM_OUT){0}; j < DIM_OUT; ++j) {
                    contexts.at(idx_m_out + j) += attention_mn*values.at(idx_n_out + j);
                }
            }
        }



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
