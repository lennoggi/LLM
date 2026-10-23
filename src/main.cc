#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <limits>

#include <cblas.h>

#include "check_parameters.hh"
#include "types.hh"
#include "declare_functions.hh"

#include "parameters.hh"


int main() {
    const auto training_text = file_to_string(TRAINING_FILE);
    const auto    input_text = file_to_string(INPUT_FILE);

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
        throw std::runtime_error("Need at least two input tokens for the cross-entropy loss calculation to be meaningful and successful");
        return 1;  // Not reached
    }


    /* Initialize a vector representation ("embedding") of each token in the
     * vocabulary with random numbers to be optimized during training */
    const auto nids_vocab = tokenizer.vocab_token2id.size();
    const auto dim_vocab  = nids_vocab*DIM;
    std::vector<double> vocab_embedding(dim_vocab);

    std::mt19937 gen;

    if constexpr (RANDOM_SEED >= 0) {
        std::cout << "INFO: seed " << RANDOM_SEED
                  << " will be used to initialize the pseudo-random number generator. The LLM output will be reproducible." << std::endl;
        gen.seed(static_cast<std::mt19937::result_type>(RANDOM_SEED));
    } else {
        std::cout << "INFO: machine entropy will be used to initialize the pseudo-random number generator. The LLM output will NOT be reproducible." << std::endl;
        std::random_device rd;
        gen.seed(rd());
    }

    std::normal_distribution<double> ndist(0.0, 1.0/std::sqrt(static_cast<double>(DIM)));

    for (auto &el : vocab_embedding) {
        el = ndist(gen);
    }


    // Initialize the positional embedding vectors with random values
    const auto dim_tot = nids_input*DIM;
    std::vector<double> pos_embeddings(dim_tot);

    for (auto &el : pos_embeddings) {
        el = ndist(gen);
    }


    /* Initialize the layer normalization scale and shift vectors to 1's and
     * and 0's, respectively
     * NOTE: one set of scale/shift vectors per application of the layer
     *       normalization:
     *         1. Before the attention block
     *         2. Before the feed-forward neural network
     *         3. Before predicting the new token */
    std::vector<double> scale_attention(DIM, 1.0), shift_attention(DIM, 0.0);
    std::vector<double>       scale_ffn(DIM, 1.0),       shift_ffn(DIM, 0.0);
    std::vector<double>     scale_final(DIM, 1.0),     shift_final(DIM, 0.0);


    // Initialize the query, key, and value weight matrices to random values
    constexpr auto dim_sq = DIM*DIM;
    std::vector<double> Wq(dim_sq), Wk(dim_sq), Wv(dim_sq);

    // Xavier/Glorot uniform distribution
    constexpr auto xg_dim_bound = std::sqrt(3.0/(static_cast<double>(DIM)));
    std::uniform_real_distribution<double> xg_dim_udist(-xg_dim_bound, xg_dim_bound);

    for (auto idx = decltype(dim_sq){0}; idx < dim_sq; ++idx) {
        Wq[idx] = xg_dim_udist(gen);
        Wk[idx] = xg_dim_udist(gen);
        Wv[idx] = xg_dim_udist(gen);
    }


    /* Initialize a uniform real distribution in [0,1] for the dropout (only
     * used if needed */
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    if constexpr (DROPOUT_PROB > 0.0) {
        std::cout << "INFO: dropout enabled with rate " << DROPOUT_PROB << std::endl;
    } else {
        std::cout << "INFO: dropout disabled" << std::endl;
    }


    /* Initialize the feed-forward neural network weights for the two layers
     * randomly, and the biases to zero
     * NOTE: think of ffn_W1 and ffn_W2 as a matrices with dimensions:
     *   - ffn_W1(DIM, DIM*FFN_EXPANSION_FACTOR)
     *   - ffn_W2(DIM*FFN_EXPANSION_FACTOR, DIM) */
    constexpr auto dim_ffn_expanded = DIM*FFN_EXPANSION_FACTOR;
    constexpr auto dim_ffn_weights  = DIM*dim_ffn_expanded;

    std::vector<double> ffn_W1(dim_ffn_weights),       ffn_W2(dim_ffn_weights);
    std::vector<double> ffn_b1(dim_ffn_expanded, 0.0), ffn_b2(DIM, 0.0);

    // Xavier/Glorot normal distribution
    constexpr auto xg_ffn_std = std::sqrt(6.0/(static_cast<double>(DIM) + static_cast<double>(dim_ffn_expanded)));
    std::normal_distribution<double> xg_ffn_ndist(0.0, xg_ffn_std);

    for (auto idx = decltype(dim_ffn_weights){0}; idx < dim_ffn_weights; ++idx) {
        ffn_W1[idx] = xg_ffn_ndist(gen);
        ffn_W2[idx] = xg_ffn_ndist(gen);
    }


    /* Initialize the logits weights to the vocabulary embedding and the biases
     * to zero
     * NOTE: think of 'logits_W' as a (DIM, nids_vocab)-shaped matrix */
    std::vector<double> logits_W(dim_vocab);
    std::vector<double> logits_b(nids_vocab, 0.0);

    for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
        const auto idx_v = v*DIM;
        for (auto i = decltype(DIM){0}; i < DIM; ++i) {
            logits_W[i*nids_vocab + v] = vocab_embedding[idx_v + i];
        }
    }


    /* Preallocate the input vectors, the query, key, and value matrices, the
     * attention and context matrices, the FFN hidden and output layers, the
     * logits vectors, the loss' gradients wrt to the model's parameters, and
     * some helpers to improve performance */
    std::vector<double> inputs(dim_tot), inputs_preFFN(dim_tot);
    std::vector<double> queries(dim_tot), keys(dim_tot), values(dim_tot);
    std::vector<double> attention(nids_input*nids_input);
    std::vector<double> context(dim_tot);

    const auto dim_tot_expanded = nids_input*dim_ffn_expanded;
    std::vector<double> ffn_h(dim_tot_expanded), ffn_h_prime(dim_tot_expanded);
    std::vector<double> ffn_out(dim_tot);

    std::vector<double> logits(nids_input*nids_vocab);

    std::vector<double> probs_m(nids_vocab);
    std::vector<double> inputs_preLN_normalized_m(DIM);
    std::vector<double> sigmas_inv_preLN(nids_input);
    std::vector<double> d_inputs_m(DIM);
    std::vector<double> dinputs_scalefinal_m(DIM);
    std::vector<double> d_ffn_b2_m(DIM);

    std::vector<double> d_ffn_b1(dim_ffn_expanded);
    std::vector<double> d_ffn_W1(dim_ffn_weights);  // Matrix (DIM, dim_ffn_expanded)

    std::vector<double> d_ffn_b2(DIM);
    std::vector<double> d_ffn_W2(dim_ffn_weights);  // Matrix (dim_ffn_expanded, DIM)

    std::vector<double> d_scale_final(DIM);
    std::vector<double> d_shift_final(DIM);

    std::vector<double> d_scale_ffn(DIM);
    std::vector<double> d_shift_ffn(DIM);

    std::vector<double> d_logits_W(dim_vocab);  // Matrix (DIM, nids_vocab)
    std::vector<double> d_logits_b(nids_vocab);


    /* Write the loss function to file at every training iteration for logging
     * purposes */
    std::ofstream loss_file(LOSS_FILE);

    if (not loss_file.is_open()) {
        throw std::runtime_error("Failed to write to file 'Loss.asc'");
        return 1;  // Not reached
    }

    loss_file << "# Column 1: training iteration" << std::endl
              << "# Column 2: loss function averaged over all input tokens" << std::endl;


    /* ========
     * Training
     * ======== */
    constexpr double sqrt_dim     = std::sqrt(static_cast<double>(DIM));
    constexpr double sqrt_dim_inv = 1.0/sqrt_dim;

    const double   loss_norm_fac = 1.0/static_cast<double>(nids_input-1);
    const double params_norm_fac = loss_norm_fac*LEARNING_RATE;

    const auto training_start = std::chrono::steady_clock::now();

    for (auto it = decltype(NTRAIN){0}; it < NTRAIN; ++it) {
        /* ************
         * Forward pass
         * ************ */
        /* --------------------------
         * Build the vector embedding
         * --------------------------
         * Map each input token ID into the corresponding embedding vector
         * (scaled by sqrt(DIM) to keep magnitudes consistent) and add the
         * corresponding positional encoding vectors */
        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            const auto idx_m    = m*DIM;
            const auto id_input = ids_input[m];
            assert(id_input < nids_vocab);
            const auto idx_vocab = id_input*DIM;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi = idx_m + i;
                inputs[mi] = sqrt_dim*vocab_embedding[idx_vocab + i] + pos_embeddings[mi];
            }
        }


        // TODO: dropout of 'inputs' without any skip connections


        /* -------------------
         * Layer normalization
         * -------------------
         * Rescale the each input embedding vector so that its components
         * average out to 0 and have variance 1, but then scale and shift them
         * by trainable parameters, to make training more stable */
        layer_norm(inputs, scale_attention, shift_attention);

        /* ------------------------------
         * Query, key, and value matrices
         * - queries = inputs*Wq
         * -    keys = inputs*Wk
         * -  values = inputs*Wv
         * ------------------------------
         * - inputs:     nids_input*DIM
         * - Wq, Wk, Wv: DIM*DIM
         * - queries, keys, values: nids_input*DIM */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, DIM, DIM, 1.0, inputs.data(), DIM, Wq.data(), DIM,
            0.0, queries.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, DIM, DIM, 1.0, inputs.data(), DIM, Wk.data(), DIM,
            0.0, keys.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, DIM, DIM, 1.0, inputs.data(), DIM, Wv.data(), DIM,
            0.0, values.data(), DIM);

        /* ----------------------------------
         * Attention matrix
         *   attention = queries * (keys^T)
         * ----------------------------------
         * - queries:   nids_input*DIM
         * - keys^T:    DIM*nids_input
         * - attention: nids_input*nids_input
         * TODO: allow for multi-head attention; need to swap
         *   nds_input<->nheads to allow parallelization by head. Then the
         *   normalization factor will become 1/sqrt(DIM_OUT/nheads) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            nids_input, nids_input, DIM, sqrt_dim_inv, queries.data(), DIM, keys.data(), DIM,
            0.0, attention.data(), nids_input);


        /* ----------------------------------------------------
         * Causal attention + softmax normalization (in one go)
         * ----------------------------------------------------
         * 1. Causal attention: zero out the upper triangular part of the
         *    attention matrix A
         * 2. Softmax normalization:  A[m,n] -> e[m,n] / sum_m(e[m,n])  with
         *                            e[m,n] := exp(A[m,n]-max(A[m,:]))
         * TODO: implement "flash attention": tile the computation of A=Q*(K^T)
         *   into smaller row blocks of Q + column blocs of K^T without storing
         *   the full attention matrix A and run an "online" softmax
         *   normalization that doesn't need a full row of A to find the row's
         *   max value */
        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            double *row    = &attention[m*nids_input];
            double row_max = -std::numeric_limits<double>::infinity();

            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                if (row[n] > row_max) {
                    row_max = row[n];
                }
            }

            // XXX: remove assertion for performance
            assert(std::isfinite(row_max));
            double sum_exp = 0.0; 

            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                row[n]   = std::exp(row[n] - row_max);
                sum_exp += row[n];
            }

            // XXX: remove assertion for performance
            assert(sum_exp > 0.0);
            const double inv_sum_exp = 1.0/sum_exp;

            for (auto n = decltype(nids_input){0}; n <= m; ++n) {
                row[n] *= inv_sum_exp;
            }

            for (auto n = decltype(nids_input){m+1}; n < nids_input; ++n) {
                row[n] = 0.0;
            }
        }


        /* ----------------------------
         * Context matrix
         *   context = attention*values
         * ----------------------------
         * - attention: nids_input*nids_input
         * - values:    nids_input*DIM
         * - context    nids_input*DIM */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, DIM, nids_input, 1.0, attention.data(), nids_input, values.data(), DIM,
            0.0, context.data(), DIM);

        /* ---------------------------------------------------------------
         * Dropout plus skip/shortcut connection between the input and the
         * context
         * ---------------------------------------------------------------
         * - DROPOUT: randomly set some of the components of the context vectors
         *    to zero to avoid having the model overly rely on a few components
         *    during training. On the other hand, components which are not set
         *    to zero must be rescaled so as to keep the expectation value over
         *    the context vector constant.
         * - SHORTCUT CONNECTION: add the context vectors to the corresponding
         *    input vectors to preserve the quality of the gradient flow during
         *    the backward step */
        skip_conn_dropout(inputs, context, udist, gen);

        /* -------------------
         * Layer normalization
         * -------------------
         * NOTE: save inputs at this stage for the backward pass */
        layer_norm(inputs, scale_ffn, shift_ffn);
        inputs_preFFN = inputs;


        /* -----------------------------------------------------------
         * Expanding-contracting two-layer feed-forward neural network
         *     FFN(x) = (GELU(x*W1 + b1))*W2 + b2
         * ----------------------------------------------------------- */
        /* STEP 1: ffn_h = inputs*ffn_W1 + / [ffn_b1] \ |
                                           |   ...    | | dim_ffn_expanded times
                                           \ [ffn_b1] / |
         * - inputs: nids_input*DIM
         * - ffn_W1: DIM*(DIM*FFN_EXPANSION_FACTOR) = DIM*dim_ffn_expanded
         * - ffn_h:  nids_input*dim_ffn_expanded */
        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            std::copy(ffn_b1.begin(), ffn_b1.end(), ffn_h.begin() + m*dim_ffn_expanded);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, dim_ffn_expanded, DIM, 1.0, inputs.data(), DIM, ffn_W1.data(), dim_ffn_expanded,
            1.0, ffn_h.data(), dim_ffn_expanded);

        /* STEP 2: take (an approximation of) GELU(ffn_h)
         * NOTE: derivative of pre-GELU inputs needed for the backward pass */
        constexpr double sqrt_2_over_pi = std::sqrt(2.0/M_PI);
        constexpr double a              = 0.044715;

        for (auto idx = decltype(dim_tot_expanded){0}; idx < dim_tot_expanded; ++idx) {
            auto &x = ffn_h[idx];
            const auto ax2 = a*x*x;
            const auto sqrt_2_over_pi_x = sqrt_2_over_pi*x;
            const auto th   = tanh(sqrt_2_over_pi_x*(1.0 + ax2));
            const auto thp1 = th + 1.0;

            x *= 0.5*thp1;
            ffn_h_prime[idx] = 0.5*(thp1 + sqrt_2_over_pi_x*(1.0 + 3.0*ax2)*(1.0 - th*th));
        }

        /* STEP 3: ffn_out = ffn_h*ffn_W2 + / [ffn_b2] \ |
                                            |   ...    | | DIM times
                                            \ [ffn_b2] / |
         * - ffn_h:  nids_input*dim_ffn_expanded
         * - ffn_W2: (DIM*FFN_EXPANSION_FACTOR)*DIM = dim_ffn_expanded*DIM
         * - ffn_out: nids_input*DIM */
        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            std::copy(ffn_b2.begin(), ffn_b2.end(), ffn_out.begin() + m*DIM);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, DIM, dim_ffn_expanded, 1.0, ffn_h.data(), dim_ffn_expanded, ffn_W2.data(), DIM,
            1.0, ffn_out.data(), DIM);


        /* --------------------------------------------------------------------
         * Dropout plus skip/shortcut connection between the network output and
         * the input
         * -------------------------------------------------------------------- */
        skip_conn_dropout(inputs, ffn_out, udist, gen);

        /* -------------------
         * Layer normalization
         * -------------------
         * NOTE: save the inverse standard deviations for each input token for
         *   the backward step. The pre-layer-norm inputs will also be needed
         *   but they will be reconstructed as they are too many to be stored
         *   while cheap to recalculate. */
        layer_norm(inputs, scale_final, shift_final, &sigmas_inv_preLN);


        /* -------------
         * Logits matrix
         * -------------
         * logits = inputs*logits_W + / [logits_b] \ |
         *                            | [logits_b] | | nids_vocab times
         *                            \ [logits_b] / |
         * - inputs:   nids_input*DIM
         * - logits_W: DIM*nids_vocab
         * - logits:   nids_input*nids_vocab */
        for (auto m = decltype(nids_input){0}; m < nids_input; ++m) {
            std::copy(logits_b.begin(), logits_b.end(), logits.begin() + m*nids_vocab);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nids_input, nids_vocab, DIM, 1.0, inputs.data(), DIM, logits_W.data(), nids_vocab,
            1.0, logits.data(), nids_vocab);


        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        /* XXX: softmax is not really needed now because it doesn't change which
         *   logit is the largest (and so it doesn't change the next predicted
         *   token). However, we'll later introduce some variability, whereby
         *   the next predicted token is not the one corresponding to the
         *   largest logit, and softmax will be useful then. */
        //softmax(last_logits);  // XXX: last_logits_defined below
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX



        /* *************
         * Backward pass
         * ************* */
        /* Compute the cross-entropy loss between the input and the target
         * tokens, where the "target" token of each input token is just the next
         * input token. Meanwhile, accumulate the terms needed to later
         * calculate the gradients of the loss wrt the logits' weights and
         * biases. */
        double loss = 0.0;

        fill(d_ffn_b1.begin(), d_ffn_b1.end(), 0.0);
        fill(d_ffn_W1.begin(), d_ffn_W1.end(), 0.0);

        fill(d_ffn_b2.begin(), d_ffn_b2.end(), 0.0);
        fill(d_ffn_W2.begin(), d_ffn_W2.end(), 0.0);

        fill(d_scale_final.begin(), d_scale_final.end(), 0.0);
        fill(d_shift_final.begin(), d_shift_final.end(), 0.0);

        fill(d_scale_ffn.begin(), d_scale_ffn.end(), 0.0);
        fill(d_shift_ffn.begin(), d_shift_ffn.end(), 0.0);

        fill(d_logits_b.begin(), d_logits_b.end(), 0.0);
        fill(d_logits_W.begin(), d_logits_W.end(), 0.0);


        for (auto m = decltype(nids_input){0}; m < nids_input - 1; ++m) {
            const auto idx_m       = m*DIM;
            const auto idx_m_vocab = m*nids_vocab;
            const auto idx_m_exp   = m*dim_ffn_expanded;

            // Find the largest logit for the current input token
            double logits_m_max = -std::numeric_limits<double>::infinity();

            for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                const auto logits_mv = logits.at(idx_m_vocab + v);
                if (logits_mv > logits_m_max) {
                    logits_m_max = logits_mv;
                }
            }

            /* Build the log of the sum of the stabilized exponentials of all
             * the logits for the current input token */
            double sum_exp_m = 0.0;

            for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                const auto exp_term = std::exp(logits.at(idx_m_vocab + v) - logits_m_max);
                probs_m.at(v) = exp_term;  // NOTE: not yet normalized by sum_exp_m
                sum_exp_m    += exp_term;
            }

            assert(sum_exp_m > 0.0);
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
             *  which implicitly applies softmax normalization to each logit to
             *  convert it into a probability */
            const auto next_input_id = ids_input.at(m+1);
            loss += -(logits.at(idx_m_vocab + next_input_id) - logits_m_max) + log_sum_exp_m;


            /* Normalize the softmax probabilities for each logit in the logits
             * vector for the current input token (i.e., for the current m
             * index) and accumulate the loss' gradients wrt the logits' weights
             * and biases and wrt the final inputs */
            fill(d_inputs_m.begin(), d_inputs_m.end(), 0.0);

            for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
                auto probs_mv = probs_m.at(v);
                probs_mv     /= sum_exp_m;

                if (v == next_input_id) {
                    probs_mv -= 1.0;
                }

                d_logits_b.at(v) += probs_mv;

                for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                    const auto iv = i*nids_vocab + v;
                    d_logits_W.at(iv) += probs_mv*inputs.at(idx_m + i);
                    d_inputs_m.at(i)  += probs_mv*logits_W.at(iv);
                }
            }


            /* Recover the pre-final-layer-norm input vector (shifted by its
             * mean and scaled by its standard deviation) for the current input
             * token. Meanwhile, accumulate the loss' gradient wrt the post-FFN
             * scale and shift.
             * NOTE: stabilize if scale_final is too small */
            auto dinputs_scalefinal_m_sum             = 0.0;
            auto dinputs_scalefinal_inputspreLN_m_sum = 0.0;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto d_inputs_mi   = d_inputs_m.at(i);
                const auto scale_final_i = scale_final.at(i);

                const auto input_preLN_normalized_mi = (scale_final_i > TOLERANCE) ?
                    (inputs.at(idx_m + i) - shift_final.at(i))/scale_final_i : 0.0;
                inputs_preLN_normalized_m.at(i) = input_preLN_normalized_mi;

                d_shift_final.at(i) += d_inputs_mi;
                d_scale_final.at(i) += d_inputs_mi*input_preLN_normalized_mi;

                const auto dinputs_scalefinal_mi      = d_inputs_mi*scale_final_i;
                dinputs_scalefinal_m.at(i)            = dinputs_scalefinal_mi;
                dinputs_scalefinal_m_sum             += dinputs_scalefinal_mi;
                dinputs_scalefinal_inputspreLN_m_sum += dinputs_scalefinal_mi*input_preLN_normalized_mi;
            }


            /* Build the loss gradients wrt to the FFN weights and biases and
             * the pre-FFN layer normalization scale and shift */
            const auto sigma_inv_preLN_m = sigmas_inv_preLN.at(m);

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto idx_i_exp = i*dim_ffn_expanded;

                const auto d_ffn_b2_mi = (dinputs_scalefinal_m.at(i)
                    - (dinputs_scalefinal_m_sum + dinputs_scalefinal_inputspreLN_m_sum*inputs_preLN_normalized_m.at(i))/static_cast<double>(DIM)
                    )*sigma_inv_preLN_m;

                d_ffn_b2_m.at(i) = d_ffn_b2_mi;
                d_ffn_b2.at(i)  += d_ffn_b2_mi;

                for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
                    d_ffn_W2.at(r*DIM + i) += d_ffn_b2_mi*ffn_h.at(idx_m_exp + r);
                }

                const auto scale_ffn_i = scale_ffn.at(i);
                const auto inputs_preFFN_normalized_mi = (scale_ffn_i > TOLERANCE) ?
                    (inputs_preFFN.at(idx_m + i) - shift_ffn.at(i))/scale_ffn_i : 0.0;

                d_shift_ffn.at(i) += d_ffn_b2_mi;
                d_scale_ffn.at(i) += d_ffn_b2_mi*inputs_preFFN_normalized_mi;
            }

            for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
                const auto idx_r  = r*DIM;
                double d_ffn_b1_r = 0.0;

                for (auto j = decltype(DIM){0}; j < DIM; ++j) {
                    d_ffn_b1_r += d_ffn_b2_m.at(j)*ffn_W2.at(idx_r + j)*ffn_h_prime.at(idx_m_exp + r);
                }

                d_ffn_b1.at(r) += d_ffn_b1_r;

                for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                    d_ffn_W1.at(i*dim_ffn_expanded + r) += d_ffn_b1_r*inputs_preFFN.at(idx_m + i);
                }
            }
        }


        // Compute average loss and update the model's parameters
        loss *= loss_norm_fac;
        loss_file << it << "\t" << loss << std::endl;

        for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
            ffn_b1.at(r) -= params_norm_fac*d_ffn_b1.at(r);
        }

        for (auto i = decltype(DIM){0}; i < DIM; ++i) {
            ffn_b2.at(i)      -= params_norm_fac*d_ffn_b2.at(i);
            scale_final.at(i) -= params_norm_fac*d_scale_final.at(i);
            shift_final.at(i) -= params_norm_fac*d_shift_final.at(i);
            scale_ffn.at(i)   -= params_norm_fac*d_scale_ffn.at(i);
            shift_ffn.at(i)   -= params_norm_fac*d_shift_ffn.at(i);
        }

        for (auto idx = decltype(dim_ffn_weights){0}; idx < dim_ffn_weights; ++idx) {
            ffn_W1.at(idx) -= params_norm_fac*d_ffn_W1.at(idx);
            ffn_W2.at(idx) -= params_norm_fac*d_ffn_W2.at(idx);
        }

        for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
            logits_b.at(v) -= params_norm_fac*d_logits_b.at(v);
        }

        for (auto idx = decltype(dim_vocab){0}; idx < dim_vocab; ++idx) {
            logits_W.at(idx) -= params_norm_fac*d_logits_W.at(idx);
        }

        // TODO: update all the other weights in the model

        if constexpr (VERBOSE) {
            std::cout << "Training epoch " << it << " completed" << std::endl;
        }

        // XXX
        // Predict the next token
        //const auto last_token_idx = nids_input*nids_vocab - nids_vocab;  // (nids_input-1)*nids_vocab
        //std::vector<double> last_logits(nids_vocab);
        //for (auto v = decltype(nids_vocab){0}; v < nids_vocab; ++v) {
        //    last_logits.at(v) = logits.at(last_token_idx + v);
        //}
        //const auto new_id    = distance(last_logits.begin(), max_element(last_logits.begin(), last_logits.end()));
        //const auto new_token = tokenizer.decode(std::vector<size_t>{new_id});
        //std::cout << input_text << " " << new_token << std::endl;
        // XXX
    }


    const auto training_stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> training_time_s = training_stop - training_start;

    std::cout << "INFO: all done. Training took " << training_time_s.count() << " s." << std::endl;



    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    //for (const auto &token : tokens) {
    //    std::cout << token << " ";
    //} std::cout << std::endl;

    //std::cout << std::endl << "***** Decoded text *****" << std::endl;
    //const auto &input_text_decoded = tokenizer.decode(tokens);
    //std::cout << input_text_decoded << std::endl;
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX


    // TODO: generate input-output token ID pairs (input must have size CONTEXT_SIZE)
    //if (CONTEXT_SIZE >= nids) {
    //    std::cerr << "Context size (" << CONTEXT_SIZE << ") must be smaller than the number of tokens in the training text (" << nids << ")" << std::endl;
    //    return 1;
    //} else {
    //    const auto n_io = nids - CONTEXT_SIZE + 1;
    //    assert(n_io > 0);

    //    std::vector<array<size_t, CONTEXT_SIZE + 1>> ios(n_io);

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
    //    //        std::cout << token << " ";
    //    //    }
    //    //    std::cout << "\t" << output << std::endl;
    //    //}
    //    // XXX
    //}

    return 0;
}
