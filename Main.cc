#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <random>

#include "Check_parameters.hh"
#include "Types.hh"
#include "include/Declare_functions.hh"

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

        // Encode the input text into token IDs
        const auto &ids_input = tokenizer.encode(input_text);
        const auto nids_input = ids_input.size();


        /* Initialize a vector representation ("embedding") of each token in the
         * vocabulary with random numbers (to be optimized during training
         * later on)                                                            */
        const auto nids_vocab = tokenizer.vocab_token2id.size();
        vector<double> vocab_embedding(nids_vocab*DIM);

        random_device rd;
        #if (RANDOM_SEED > 0)
        cout << "INFO: seed " << RANDOM_SEED
             << " will be used to initialize the pseudo-random number generator. The LLM output will be reproducible." << endl;
        mt19937 gen(RANDOM_SEED);
        #else
        cout << "INFO: machine entropy will be used to initialize the pseudo-random number generator. The LLM output will NOT be reproducible." << endl;
        mt19937 gen(rd());
        #endif
        normal_distribution<double> ndist(0., 1./sqrt(static_cast<double>(DIM)));

        for (auto &el : vocab_embedding) {
            el = ndist(gen);
        }


        // Initialize the positional embedding vectors with random numbers
        const auto dim_tot = nids_input*DIM;
        vector<double> pos_embeddings(dim_tot);

        for (auto &el : pos_embeddings) {
            el = ndist(gen);
        }


        /* Initialize the layer normalization scale and shift vectors to 1's
         * and 0's, respectively
         * NOTE: one set of scale/shift vectors per application of the layer
         *       normalization:
         *         1. Before the attention block
         *         2. Before the feed-forward neural network
         *         3. Before predicting the new token                           */
        vector<double> scale_attention(DIM, 1.), shift_attention(DIM, 0.);
        vector<double>       scale_ffn(DIM, 1.),       shift_ffn(DIM, 0.);
        // /*TODO*/ vector<double> scale_final(DIM, 1.), shift_final(DIM, 0.);


        /* Initialize the query, key, and value weight matrices to random values
         * NOTE: declared as std::vector so they are allocated on the heap.
         *       std::array allocates on the stack and this can overflow for
         *       very large DIM*DIM.                                            */
        constexpr auto dim_sq = DIM*DIM;
        vector<double> Wq(dim_sq), Wk(dim_sq), Wv(dim_sq);

        // Xavier/Glorot distribution
        constexpr auto xg_bound = sqrt(3./(static_cast<double>(DIM)));
        uniform_real_distribution<double> xgdist(-xg_bound, xg_bound);

        for (auto idx = decltype(dim_sq){0}; idx < dim_sq; ++idx) {
            Wq.at(idx) = xgdist(gen);
            Wk.at(idx) = xgdist(gen);
            Wv.at(idx) = xgdist(gen);
        }


        /* Initialize a uniform real distribution in [0,1] for the dropout (only
         * used if needed                                                       */
        uniform_real_distribution<double> udist(0., 1.);

        if constexpr (DROPOUT_PROB > 0.) {
            cout << "INFO: dropout enabled with rate " << DROPOUT_PROB << endl;
        } else {
            cout << "INFO: dropout disabled" << endl;
        }



        /* ========
         * Training
         * ======== */
        for (auto it = decltype(NTRAIN){0}; it < NTRAIN; ++it) {
            /* ------------
             * Forward pass
             * ------------ */
            /* Map each input token ID into the corresponding embedding vector
             * (scaled by sqrt(DIM) to keep magnitudes consistent) and add the
             * corresponding positional encoding vectors                        */
            vector<double> inputs(dim_tot);
            constexpr auto sqrt_dim = sqrt(static_cast<double>(DIM));

            for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
                const auto idx_m = m*DIM;
                const auto id_input = ids_input.at(m);
                assert(id_input < nids_vocab);
                const auto idx_vocab = id_input*DIM;

                for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                    const auto mi = idx_m + i;
                    inputs.at(mi) = sqrt_dim*vocab_embedding.at(idx_vocab + i) + pos_embeddings.at(mi);
                }
            }


            /* Layer normalization: have the components of each input embedding
             * vector average out to 0 and have variance 1, but then scale and
             * shift them by trainable parameters, to make training more stable */
            layer_norm(inputs, scale_attention, shift_attention);


            // Build the query, key, and value matrices
            vector<double> queries(dim_tot, 0.), keys(dim_tot, 0.), values(dim_tot, 0.);  // NOTE: initialize to zero

            for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
                const auto idx_m = m*DIM;

                /* NOTE: swapping the more "natural" loop order (j out, k in) to
                 *       improve the memory access pattern in Wq, Wk, Wv        */
                for (auto k = decltype(DIM){0}; k < DIM; ++k) {
                    const auto inputs_mk = inputs.at(idx_m + k);
                    const auto idx_k     = k*DIM;

                    for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                        const auto mi = idx_m + i;
                        const auto ki = idx_k + i;

                        queries.at(mi) += inputs_mk*Wq.at(ki);
                           keys.at(mi) += inputs_mk*Wk.at(ki);
                         values.at(mi) += inputs_mk*Wv.at(ki);
                    }
                }
            }


            /* Compute the attention scores and the context vectors (matrix),
             * i.e., the sum of the value vectors (columns of the values matrix)
             * weighted by the attention scores along the rows of the attention
             * matrix
             * NOTE: no need to store the full attention matrix: only compute
             *       each row and the corresponding context vector              */
            /* TODO: allow for multi-head attention; need to swap
             *   nds_input<->nheads to allow parallelization by head. Then the
             *   normalization factor will become 1/sqrt(DIM_OUT/nheads)        */
            constexpr auto sqrt_dim_inv = 1./sqrt_dim;
            vector<double> contexts(dim_tot);

            for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
                const auto idx_m = m*DIM;

                /* Causal attention: each token ID in the input text only attends
                 * to all the previous ones, so that the attention scores in the
                 * upper triangular part of the attention scores matrix (i.e.,
                 * all the attention scores for n > m for row/token m) are zero
                 * (not even defined here)                                      */
                vector<double> attention_m(m+1);  // Instead of attention_m(nids)

                for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                    const auto idx_n = n*DIM;
                    double attention_mn = 0.;

                    for (auto l = decltype(DIM){0}; l < DIM; ++l) {
                        const auto ml = idx_m + l;
                        const auto nl = idx_n + l;
                        attention_mn += queries.at(ml)*keys.at(nl);
                    }

                    /* Scale the attention score by
                     * sqrt(len(keys[:,1]) = sqrt(DIM) to improve training
                     * behavior later on                                        */
                    attention_m.at(n) = attention_mn*sqrt_dim_inv;
                }


                /* Normalize the attention scores for the current token (i.e.,
                 * for the current row of the attention matrix) using a
                 * stabilized softmax                                           */
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


                /* Finish normalizing the attention scores for the current token
                 * (i.e., for the current row of the attention matrix) and
                 * calculate the context vector for the current token
                 * NOTE: swapping the more "natural" loop order (j out, n in) to
                 *       improve the memory access pattern in the values matrix */
                for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                    const auto idx_n = n*DIM;
                    const auto attention_mn = attention_m.at(n)/sum_exp;  // NOTE: attention_m.at(n) is NOT normalized

                    for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                        contexts.at(idx_m + i) += attention_mn*values.at(idx_n + i);
                    }
                }
            }


            /* ***** Dropout + Shortcut connection *****
             * Dropout: randomly set some of the components of the context
             *   vectors to zero to avoid having the model overly rely on a few
             *   components during training. On the other hand, components which
             *   are not set to zero must be rescaled so as to keep the
             *   expectation value over the context vector constant.
             * Shortcut connection: add the context vectors to the corresponding
             *   input vectors to preserve the quality of the gradient flow
             *   during the backward step
             * NOTE: this could be moved to into the attention loop (m loop)
             *       above, but the plain loop here is more straightforward     */
            for (auto idx = decltype(dim_tot){0}; idx < dim_tot; ++idx) {
                if constexpr (DROPOUT_PROB > 0.) {
                    const auto x = udist(gen);
                    if (x < DROPOUT_PROB) {
                        contexts.at(idx) = 0.;
                    } else {
                        contexts.at(idx) *= 1./(1. - DROPOUT_PROB);
                    }
                }

                inputs.at(idx) += contexts.at(idx);
            }


            // Another layer normalization
            layer_norm(inputs, scale_ffn, shift_ffn);


            // TODO: feed-forward NN with 4X expansion and contraction, dropout again


            // TODO: layer normalization


            // TODO: predict the next token (include max_new token and context size)



            /* -------------------
             * TODO: backward pass
             * ------------------- */

            // XXX
            cout << "Training epoch " << it << " completed" << endl;
            // XXX
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
