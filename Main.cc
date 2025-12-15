#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <limits>

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

        if (nids_input < 2) {
            throw runtime_error("Need at least two input tokens for the cross-entropy loss calculation to be meaningful and successful");
            return 1;  // Not reached
        }


        /* Initialize a vector representation ("embedding") of each token in the
         * vocabulary with random numbers (to be optimized during training
         * later on)                                                            */
        const auto nids_vocab = tokenizer.vocab_token2id.size();
        const auto dim_vocab  = nids_vocab*DIM;
        vector<double> vocab_embedding(dim_vocab);

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
        vector<double>     scale_final(DIM, 1.),     shift_final(DIM, 0.);


        /* Initialize the query, key, and value weight matrices to random values
         * NOTE: declared as std::vector so they are allocated on the heap.
         *       std::array allocates on the stack and this can overflow for
         *       very large DIM*DIM.                                            */
        constexpr auto dim_sq = DIM*DIM;
        vector<double> Wq(dim_sq), Wk(dim_sq), Wv(dim_sq);

        // Xavier/Glorot uniform distribution
        constexpr auto xg_dim_bound = sqrt(3./(static_cast<double>(DIM)));
        uniform_real_distribution<double> xg_dim_udist(-xg_dim_bound, xg_dim_bound);

        for (auto idx = decltype(dim_sq){0}; idx < dim_sq; ++idx) {
            Wq.at(idx) = xg_dim_udist(gen);
            Wk.at(idx) = xg_dim_udist(gen);
            Wv.at(idx) = xg_dim_udist(gen);
        }


        /* Initialize a uniform real distribution in [0,1] for the dropout (only
         * used if needed                                                       */
        uniform_real_distribution<double> udist(0., 1.);

        if constexpr (DROPOUT_PROB > 0.) {
            cout << "INFO: dropout enabled with rate " << DROPOUT_PROB << endl;
        } else {
            cout << "INFO: dropout disabled" << endl;
        }


        /* Initialize the feed-forward neural network weights for the two layers
         * randomly, and the biases to zero
         * NOTE: think of ffn_W1 and ffn_W2 as a matrices with dimensions:
         *   - ffn_W1(DIM, DIM*FFN_EXPANSION_FACTOR)
         *   - ffn_W2(DIM*FFN_EXPANSION_FACTOR, DIM)                            */
        constexpr auto dim_ffn_expanded = DIM*FFN_EXPANSION_FACTOR;
        constexpr auto dim_ffn_weights  = DIM*dim_ffn_expanded;

        vector<double> ffn_W1(dim_ffn_weights),      ffn_W2(dim_ffn_weights);
        vector<double> ffn_b1(dim_ffn_expanded, 0.), ffn_b2(DIM, 0.);

        // Xavier/Glorot normal distribution
        constexpr auto xg_ffn_std = sqrt(6./(static_cast<double>(DIM) + static_cast<double>(dim_ffn_expanded)));
        normal_distribution<double> xg_ffn_ndist(0., xg_ffn_std);

        for (auto idx = decltype(dim_ffn_weights){0}; idx < dim_ffn_weights; ++idx) {
            ffn_W1.at(idx) = xg_ffn_ndist(gen);
            ffn_W2.at(idx) = xg_ffn_ndist(gen);
        }


        /* Initialize the logits weights to the vocabulary embedding and the
         * biases to zero
         * NOTE: think of 'logits_W' as a (DIM, nids_vocab)-shaped matrix       */
        vector<double> logits_W(dim_vocab);
        vector<double> logits_b(nids_vocab, 0.);

        for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
            const auto idx_v = v*DIM;
            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                logits_W.at(i*nids_vocab + v) = vocab_embedding.at(idx_v + i);
            }
        }


        /* Preallocate the input and context vectors, the query, key, and value
         * matrices, and the FFN hidden and output layers, the logits vectors,
         * the loss' gradients wrt to the logits weights and biases, and some
         * helpers to improve performance                                       */
        vector<double> inputs(dim_tot), contexts(dim_tot);
        vector<double> queries(dim_tot), keys(dim_tot), values(dim_tot);

        const auto dim_tot_expanded = nids_input*dim_ffn_expanded;
        vector<double> ffn_h(dim_tot_expanded);
        vector<double> ffn_out(dim_tot);

        vector<double> logits(nids_input*nids_vocab);
        vector<double> probs_m(nids_vocab);
        vector<double> d_logits_W(dim_vocab);  // Matrix (DIM, nids_vocab)
        vector<double> d_logits_b(nids_vocab);



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


            // TODO: dropout of 'inputs' without any skip connections


            /* Layer normalization: have the components of each input embedding
             * vector average out to 0 and have variance 1, but then scale and
             * shift them by trainable parameters, to make training more stable */
            layer_norm(inputs, scale_attention, shift_attention);


            // Build the query, key, and value matrices
            fill(queries.begin(), queries.end(), 0.);
            fill(   keys.begin(),    keys.end(), 0.);
            fill( values.begin(),  values.end(), 0.);

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
            fill(contexts.begin(), contexts.end(), 0.);

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
                softmax(attention_m);

                /* Calculate the context vector for the current token
                 * NOTE: swapping the more "natural" loop order (j out, n in) to
                 *       improve the memory access pattern in the values matrix */
                for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                    const auto idx_n = n*DIM;
                    const auto attention_mn = attention_m.at(n);

                    for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                        contexts.at(idx_m + i) += attention_mn*values.at(idx_n + i);
                    }
                }
            }


            /* *Dropout:* randomly set some of the components of the context
             *   vectors to zero to avoid having the model overly rely on a few
             *   components during training. On the other hand, components which
             *   are not set to zero must be rescaled so as to keep the
             *   expectation value over the context vector constant.
             * *Shortcut connection:* add the context vectors to the
             *   corresponding input vectors to preserve the quality of the
             *   gradient flow during the backward step                         */
            skip_conn_dropout(inputs, contexts, udist, gen);

            // Another layer normalization
            layer_norm(inputs, scale_ffn, shift_ffn);


            /* Expanding-contracting two-layer feed-forward neural network:
             *   FFN(x) = (GELU(x*W1 + b1))*W2 + b2                             */
            for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
                const auto idx_m     = m*DIM;
                const auto idx_m_exp = m*dim_ffn_expanded;

                for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
                    ffn_h.at(idx_m_exp + r) = ffn_b1.at(r);
                }

                for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                    const auto inputs_mi = inputs.at(idx_m + i);
                    const auto idx_i_exp = i*dim_ffn_expanded;

                    for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
                        ffn_h.at(idx_m_exp + r) += inputs_mi*ffn_W1.at(idx_i_exp + r);
                    }
                }
            }

            // Not quite GELU, just an approximation
            GELU_approx(ffn_h);

            for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
                const auto idx_m     = m*DIM;
                const auto idx_m_exp = m*dim_ffn_expanded;

                for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                    ffn_out.at(idx_m + i) = ffn_b2.at(i);
                }

                for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
                    const auto ffn_h_mr = ffn_h.at(idx_m_exp + r);
                    const auto idx_r    = r*DIM;

                    for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                        ffn_out.at(idx_m + i) += ffn_h_mr*ffn_W2.at(idx_r + i);
                    }
                }
            }


            /* Apply dropout (if enabled) to the network's output and set up a
             * skip connection between that and the input vectors               */
            skip_conn_dropout(inputs, ffn_out, udist, gen);


            // Another layer normalization
            layer_norm(inputs, scale_final, shift_final);


            // Build the logits vector for each input token
            for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
                const auto idx_m       = m*DIM;
                const auto idx_m_vocab = m*nids_vocab;

                for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                    logits.at(idx_m_vocab + v) = logits_b.at(v);
                }

                for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                    const auto idx_i_vocab = i*nids_vocab;
                    const auto inputs_mi   = inputs.at(idx_m + i);

                    for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                        logits.at(idx_m_vocab + v) += inputs_mi*logits_W.at(idx_i_vocab + v);
                    }
                }
            }

            // XXX XXX XXX XXX XXX XXX
            // XXX XXX XXX XXX XXX XXX
            // XXX XXX XXX XXX XXX XXX
            /* XXX: softmax is not really needed now because it doesn't change
             *      which logit is the largest (and so it doesn't change the
             *      next predicted token). However, we'll later introduce some
             *      variability, whereby the next predicted token is not
             *      the one corresponding to the largest logit, and softmax will
             *      be useful then.                                             */
            //softmax(last_logits);  // XXX: last_logits_defined below
            // XXX XXX XXX XXX XXX XXX
            // XXX XXX XXX XXX XXX XXX
            // XXX XXX XXX XXX XXX XXX


            // XXX
            // Predict the next token
            //const auto last_token_idx = nids_input*nids_vocab - nids_vocab;  // (nids_input-1)*nids_vocab
            //vector<double> last_logits(nids_vocab);
            //for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
            //    last_logits.at(v) = logits.at(last_token_idx + v);
            //}
            //const auto new_id    = distance(last_logits.begin(), max_element(last_logits.begin(), last_logits.end()));
            //const auto new_token = tokenizer.decode(vector<size_t>{new_id});
            //cout << input_text << endl << new_token << endl;
            // XXX



            /* -------------------
             * TODO: backward pass
             * ------------------- */
            /* Compute the cross-entropy loss between the input and the target
             * tokens, where the "target" token of each input token is just the
             * next input token. Meanwhile, accumulate the terms needed to later
             * calculate the gradients of the loss wrt the logits' weights and
             * biases.                                                          */
            double loss = 0.;
            fill(d_logits_b.begin(), d_logits_b.end(), 0.);
            fill(d_logits_W.begin(), d_logits_W.end(), 0.);

	    for (auto m = decltype(nids_input){0}; m < nids_input - 1; ++m) {
                const auto idx_m_vocab =  m*nids_vocab;

                // Find the largest logit for the current input token
                double logits_m_max = -numeric_limits<double>::infinity();

                for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                    const auto logits_mv = logits.at(idx_m_vocab + v);
                    if (logits_mv > logits_m_max) {
                        logits_m_max = logits_mv;
                    }
                }

                /* Build the log of the sum of the stabilized exponentials of
                 * all the logits for the current input token                   */
                double sum_exp_m = 0.;

                for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                    const auto exp_term = exp(logits.at(idx_m_vocab + v) - logits_m_max);
                    probs_m.at(v) = exp_term;  // NOTE: not yet normalized by sum_exp_m
                    sum_exp_m    += exp_term;
                }

                assert(sum_exp_m > 0.);
                const auto log_sum_exp_m = log(sum_exp_m);

                /* Add the loss term for the current input token
                 * NOTE: letting
                 *   z[m][v+1] = logits.at(idx_m_vocab + ids_input.at(m+1))
                 *   be the logit corresponding to the next input token, the
                 *   expression below is equivalent to
                 *
                 *       -log(exp(z[m][v+1] - logits_m_max)) + log_sum_exp_m  =
                 *   = -( log(exp(z[m][v+1] - logits_m_max)) - log_sum_exp_m) =
                 *   = -log( (exp(z[m][v+1] - logits_m_max)) / sum_exp_m) ,
                 *
                 *  which implicitly applies softmax normalization to each logit
                 *  to convert it into a probability                            */
                const auto next_input_id = ids_input.at(m+1);
                loss += -(logits.at(idx_m_vocab + next_input_id) - logits_m_max) + log_sum_exp_m;


                /* Normalize the softmax probabilities for each logit in the
                 * logits vector for the current input token (i.e., for the
                 * current m index) and accumulate the loss' gradients wrt the
                 * logits' weights and biases                                   */
                const auto idx_m = m*DIM;

                for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                    auto probs_mv = probs_m.at(v);
                    probs_mv     /= sum_exp_m;

                    if (v == next_input_id) {
                        probs_mv -= 1.;
                    }

                    d_logits_b.at(v) += probs_mv;

                    for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                        const auto idx_i_vocab = i*nids_vocab;
                        d_logits_W.at(i*nids_vocab + v) += probs_mv*inputs.at(idx_m + i);
                    }
                }
            }


            // Compute average loss and update the logits weights and biases
            auto norm_fac = 1./static_cast<double>(nids_input-1);
            loss         *= norm_fac;
            norm_fac     *= LEARNING_RATE;

            for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                logits_b.at(v) -= norm_fac*d_logits_b.at(v);
            }

            for (auto idx = decltype(dim_vocab){0}; idx < dim_vocab; ++idx) {
                logits_W.at(idx) -= norm_fac*d_logits_W.at(idx);
            }


            // TODO: update all the other weights in the model

            cout << "Training epoch " << it << " completed, average loss: " << loss << endl;
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
