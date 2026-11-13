#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#include <cblas.h>

#include "check_parameters.hh"
#include "types.hh"
#include "declare_functions.hh"

#include "parameters.hh"


int main() {
    /* -------------
     * Text encoding
     * ------------- */
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
    const auto nx = ids_input.size();

    if (nx < 2) {
        throw std::runtime_error("Need at least two input tokens for the cross-entropy loss calculation to be meaningful and successful");
        return 1;  // Not reached
    }

    const auto nv = tokenizer.vocab_token2id.size();

    // Matrix dimensions used repeatedly
    constexpr auto dim_sq           = DIM*DIM;
    constexpr auto dim_ffn_expanded = DIM*FFN_EXPANSION_FACTOR;
    constexpr auto dim_ffn_weights  = DIM*dim_ffn_expanded;
    const     auto nxdim            = nx*DIM;
    const     auto nvdim            = nv*DIM;
    const     auto nxnv             = nx*nv;


    /* -------------------------------
     * Initialize learnable parameters
     * ------------------------------- */
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


    /* Initialize a vector representation ("embedding") of each token in the
     * vocabulary with random numbers to be optimized during training */
    std::vector<double> vocab_embeddings(nvdim);

    for (auto &el : vocab_embeddings) {
        el = ndist(gen);
    }


    // Initialize the positional embedding vectors with random values
    std::vector<double> pos_embeddings(nxdim);

    for (auto &el : pos_embeddings) {
        el = ndist(gen);
    }


    /* Initialize the three layer normalization (LN) scale and shift vectors to
     * 1's and 0's, respectively. The three LN steps are: */
    std::vector<double> scale1(DIM, 1.0), shift1(DIM, 0.0);  // LN before the attention block
    std::vector<double> scale2(DIM, 1.0), shift2(DIM, 0.0);  // LN before the feed-forward neural network (FFN)
    std::vector<double> scale3(DIM, 1.0), shift3(DIM, 0.0);  // LN before predicting the new token


    // Initialize the query, key, and value weight matrices with random values
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
     * to zero */
    std::vector<double> logits_W(nvdim);  // Matrix (DIM, nv)
    std::vector<double> logits_b(nv, 0.0);

    for (auto v = decltype(nv){0}; v < nv; ++v) {
        const auto idx_v = v*DIM;
        for (auto i = decltype(DIM){0}; i < DIM; ++i) {
            logits_W[i*nv + v] = vocab_embeddings[idx_v + i];
        }
    }


    /* ------------------
     * Preallocate memory
     * ------------------ */
    /* - Input vectors `x`, which change along the way during the forward
     *   pass due to layer normalization and skip connections
     * - Input vectors `x1` after LN1
     * - Input vectors `x2` after LN2 */
    std::vector<double> x(nxdim), x1(nxdim), x2(nxdim);

    // Query, key, and value matrices
    std::vector<double> queries(nxdim), keys(nxdim), values(nxdim);

    // Attention and context matrices
    const size_t nxsq = nx*nx;
    std::vector<double> attention(nxsq);
    std::vector<double> context(nxdim);

    // Layer normalization intermediate parameters
    std::vector<double> xhat1(nxdim), sigmas_inv1(nx);  // LN before the attention block
    std::vector<double> xhat2(nxdim), sigmas_inv2(nx);  // LN before the feed-forward neural network (FFN)
    std::vector<double> xhat3(nxdim), sigmas_inv3(nx);  // LN before predicting the new token

    // FFN hidden and output layers
    const auto nxdim_expanded = nx*dim_ffn_expanded;
    std::vector<double> ffn_h(nxdim_expanded), ffn_h_prime(nxdim_expanded);
    std::vector<double> ffn_out(nxdim);

    // Logits vectors
    std::vector<double> logits(nxnv);

    // dL/d_logits with last row = 0
    std::vector<double> probs(nxnv);

    /* dx is overwritten several times during the backward pass:
     * 1. dx = dL/dx3
     * 2. dx = dL/dx2 [skip connection path]
     * 3. dx = dL/dx2 [total] = dL/dx2 [skip connection path] + dL/dx2 [FFN path]
     * 4. dx = dL/dx1 [total] = dL/dx1 [skip connection path] + dL/dx1 [FFN path] */
    std::vector<double> dx(nxdim);

    // dz^m_r = (dx2skip * ffn_W2^T)^m_r * ffn_h_prime^m_r
    std::vector<double> dz(nxdim_expanded);

    /* A = "attention". dA is overwritten during the backward pass:
     * 1. dA = dL/dA
     * 2. dA = dL/dS = dL/dA * dA/dS */
    std::vector<double> dA(nxsq);

    // dL/dQ (queries), dL/dK (keys), dL/dV (values)
    std::vector<double> dQ(nxdim), dK(nxdim), dV(nxdim);


    // Loss' gradients wrt to the model's parameters
    std::vector<double> dWq(dim_sq), dWk(dim_sq), dWv(dim_sq);

    std::vector<double> d_ffn_b1(dim_ffn_expanded);
    std::vector<double> d_ffn_W1(dim_ffn_weights);  // Matrix (DIM, dim_ffn_expanded)

    std::vector<double> d_ffn_b2(DIM);
    std::vector<double> d_ffn_W2(dim_ffn_weights);  // Matrix (dim_ffn_expanded, DIM)

    std::vector<double> d_scale1(DIM);
    std::vector<double> d_scale2(DIM);
    std::vector<double> d_scale3(DIM);

    std::vector<double> d_shift1(DIM);
    std::vector<double> d_shift2(DIM);
    std::vector<double> d_shift3(DIM);

    std::vector<double> d_logits_W(nvdim);  // Matrix (DIM, nv)
    std::vector<double> d_logits_b(nv);


    /* Write the loss function to file at every training iteration for logging
     * purposes */
    std::ofstream loss_file(LOSS_FILE);

    if (not loss_file.is_open()) {
        throw std::runtime_error("Failed to write to file 'Loss.asc'");
        return 1;  // Not reached
    }

    loss_file << "# Column 1: training iteration" << std::endl
              << "# Column 2: loss function averaged over all input tokens" << std::endl;


    // Print OpenBLAS configuration info
    std::cout << "INFO: OpenBLAS settings" << std::endl;
    const auto openblas_parallel_mode = openblas_get_parallel();

    if (openblas_parallel_mode == 0) {
        std::cout << "OpenBLAS parallel mode: single-threaded" << std::endl;
    } else if (openblas_parallel_mode == 1) {
        std::cout << "OpenBLAS parallel mode: pthreads or native OS threads" << std::endl;
    } else if (openblas_parallel_mode == 2) {
        std::cout << "OpenBLAS parallel mode: OpenMP threads" << std::endl;
    } else {
        std::cout << "ERROR: unexpected OpenBLAS parallel mode ("
                  << openblas_parallel_mode << ")" << std::endl;
    }   

    std::cout << "  OpenBLAS configuration: " << openblas_get_config()      << std::endl;
    std::cout << "  Core architecture:      " << openblas_get_corename()    << std::endl;
    std::cout << "  Number of threads:      " << openblas_get_num_threads() << std::endl;


    /* ========
     * Training
     * ======== */
    constexpr double dim_inv      = 1.0/static_cast<double>(DIM);
    constexpr double sqrt_dim     = std::sqrt(static_cast<double>(DIM));
    constexpr double sqrt_dim_inv = 1.0/sqrt_dim;

    const double   loss_norm_fac          = 1.0/static_cast<double>(nx-1);
    const double params_norm_fac          = loss_norm_fac*LEARNING_RATE;
    const double params_norm_fac_sqrt_dim = params_norm_fac*sqrt_dim;

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
        for (auto m = decltype(nx){0}; m < nx; ++m) {
            const auto idx_m    = m*DIM;
            const auto id_input = ids_input[m];
            assert(id_input < nv);
            const auto idx_vocab = id_input*DIM;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi = idx_m + i;
                x[mi] = sqrt_dim*vocab_embeddings[idx_vocab + i] + pos_embeddings[mi];
            }
        }


        // TODO: dropout of 'x' without any skip connections


        /* ----------------------------
         * Layer normalization #1 (LN1)
         * ----------------------------
         * Rescale the each input embedding vector so that its components
         * average out to 0 and have variance 1, but then scale and shift them
         * by trainable parameters, to make training more stable.
         *   ***** The input changes: x -> x1 ***** */
        layer_norm(x, scale1, shift1, xhat1, sigmas_inv1);
        x1 = x;

        /* ----------------
         * - queries = x*Wq
         * -    keys = x*Wk
         * -  values = x*Wv
         * ----------------
         * - x:          nx*DIM
         * - Wq, Wk, Wv: DIM*DIM
         * - queries, keys, values: nx*DIM */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, DIM, DIM, 1.0, x.data(), DIM, Wq.data(), DIM,
            0.0, queries.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, DIM, DIM, 1.0, x.data(), DIM, Wk.data(), DIM,
            0.0, keys.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, DIM, DIM, 1.0, x.data(), DIM, Wv.data(), DIM,
            0.0, values.data(), DIM);

        /* --------------------------------------------
         * attention = (1/sqrt(DIM)) * queries * keys^T
         * --------------------------------------------
         * - queries:   nx*DIM
         * - keys^T:    DIM*nx
         * - attention: nx*nx
         * TODO: allow for multi-head attention; need to swap
         *   nds_input<->nheads to allow parallelization by head. Then the
         *   normalization factor will become 1/sqrt(DIM_OUT/nheads) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            nx, nx, DIM, sqrt_dim_inv, queries.data(), DIM, keys.data(), DIM,
            0.0, attention.data(), nx);


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
        for (auto m = decltype(nx){0}; m < nx; ++m) {
                  double *attention_m = &attention[m*nx];
            /* Restrict the maximum to the causal (i.e., lower-diagonal) part
             * of the attention matrix                                          */
            const double  attention_m_max = *std::max_element(attention_m, attention_m + m + 1);
                  double  sum_exp = 0.0; 

            for (auto n = decltype(nx){0}; n <= m; ++n) {
                attention_m[n] = std::exp(attention_m[n] - attention_m_max);
                sum_exp       += attention_m[n];
            }

            const double inv_sum_exp = 1.0/sum_exp;

            for (auto n = decltype(nx){0}; n <= m; ++n) {
                attention_m[n] *= inv_sum_exp;
            }

            for (auto n = decltype(nx){m+1}; n < nx; ++n) {
                attention_m[n] = 0.0;
            }
        }


        /* ----------------------------
         * context = attention * values
         * ----------------------------
         * - attention: nx*nx
         * - values:    nx*DIM
         * - context    nx*DIM */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, DIM, nx, 1.0, attention.data(), nx, values.data(), DIM,
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
         *    the backward step.
         *   ***** The input changes: x1 -> x1skip ***** */
        skip_conn_dropout(x, context, udist, gen);

        /* ----------------------------
         * Layer normalization #2 (LN2)
         * ----------------------------
         *   ***** The input changes: x1skip -> x2 *****
         * NOTE: save the inputs at this stage to avoid rebuilding them during
         *       the backward pass */
        layer_norm(x, scale2, shift2, xhat2, sigmas_inv2);
        x2 = x;


        /* -----------------------------------------------------------
         * Expanding-contracting two-layer feed-forward neural network
         *     FFN(x) = (GELU(x*W1 + b1))*W2 + b2
         * ----------------------------------------------------------- */
        /* STEP 1: ffn_h = x*ffn_W1 + / [ffn_b1] \ |
                                      |   ...    | | dim_ffn_expanded times
                                      \ [ffn_b1] / |
         * - x:      nx*DIM
         * - ffn_W1: DIM*(DIM*FFN_EXPANSION_FACTOR) = DIM*dim_ffn_expanded
         * - ffn_h:  nx*dim_ffn_expanded */
        for (auto m = decltype(nx){0}; m < nx; ++m) {
            std::copy(ffn_b1.begin(), ffn_b1.end(), ffn_h.begin() + m*dim_ffn_expanded);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, dim_ffn_expanded, DIM, 1.0, x.data(), DIM, ffn_W1.data(), dim_ffn_expanded,
            1.0, ffn_h.data(), dim_ffn_expanded);

        /* STEP 2: take (an approximation of) GELU(ffn_h)
         * NOTE: derivative of pre-GELU inputs needed for the backward pass */
        constexpr double sqrt_2_over_pi = std::sqrt(2.0/M_PI);
        constexpr double a              = 0.044715;

        for (auto idx = decltype(nxdim_expanded){0}; idx < nxdim_expanded; ++idx) {
            auto &z = ffn_h[idx];
            const auto az2 = a*z*z;
            const auto sqrt_2_over_pi_z = sqrt_2_over_pi*z;
            const auto th   = tanh(sqrt_2_over_pi_z*(1.0 + az2));
            const auto thp1 = th + 1.0;

            z *= 0.5*thp1;
            ffn_h_prime[idx] = 0.5*(thp1 + sqrt_2_over_pi_z*(1.0 + 3.0*az2)*(1.0 - th*th));
        }

        /* STEP 3: ffn_out = ffn_h*ffn_W2 + / [ffn_b2] \ |
                                            |   ...    | | DIM times
                                            \ [ffn_b2] / |
         * - ffn_h:  nx*dim_ffn_expanded
         * - ffn_W2: (DIM*FFN_EXPANSION_FACTOR)*DIM = dim_ffn_expanded*DIM
         * - ffn_out: nx*DIM */
        for (auto m = decltype(nx){0}; m < nx; ++m) {
            std::copy(ffn_b2.begin(), ffn_b2.end(), ffn_out.begin() + m*DIM);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, DIM, dim_ffn_expanded, 1.0, ffn_h.data(), dim_ffn_expanded, ffn_W2.data(), DIM,
            1.0, ffn_out.data(), DIM);


        /* --------------------------------------------------------------------
         * Dropout plus skip/shortcut connection between the network output and
         * the input
         * --------------------------------------------------------------------
         *   ***** The input changes: x2 -> x2skip ***** */
        skip_conn_dropout(x, ffn_out, udist, gen);

        /* ----------------------------
         * Layer normalization #3 (LN3)
         * ---------------------------- */
        layer_norm(x, scale3, shift3, xhat3, sigmas_inv3);


        /* ------------------------------------------------
         * logits = x*logits_W + / [logits_b] \ |
         *                       | [logits_b] | | nv times
         *                       \ [logits_b] / |
         * ------------------------------------------------
         * - x:        nx*DIM
         * - logits_W: DIM*nv
         * - logits:   nx*nv */
        for (auto m = decltype(nx){0}; m < nx; ++m) {
            std::copy(logits_b.begin(), logits_b.end(), logits.begin() + m*nv);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            nx, nv, DIM, 1.0, x.data(), DIM, logits_W.data(), nv,
            1.0, logits.data(), nv);


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
        fill(d_ffn_b2.begin(), d_ffn_b2.end(), 0.0);

        fill(d_scale3.begin(), d_scale3.end(), 0.0);
        fill(d_shift3.begin(), d_shift3.end(), 0.0);

        fill(d_scale2.begin(), d_scale2.end(), 0.0);
        fill(d_shift2.begin(), d_shift2.end(), 0.0);

        fill(d_logits_b.begin(), d_logits_b.end(), 0.0);


        /* ------------------------------------------------------
         * 1. Compute the loss function and build
         *    probs^m_v = dL^m / dlogits^m_v =
         *  = softmax(logits)^m_v - delta_{v, v_next_input_token}
         * ------------------------------------------------------ */
        for (auto m = decltype(nx){0}; m < nx - 1; ++m) {
            const auto idx_m_vocab = m*nv;

                  double *logits_m     = &logits[idx_m_vocab];
            const double  logits_m_max = *std::max_element(logits_m, logits_m + nv);
                  double  sum_exp_m    = 0.0;

            for (auto v = decltype(nv){0}; v < nv; ++v) {
                const auto mv       = idx_m_vocab + v;
                const auto exp_term = std::exp(logits[mv] - logits_m_max);
                probs[mv]  = exp_term;  // NOTE: not yet normalized by sum_exp_m
                sum_exp_m += exp_term;
            }

            /* Add the loss term for the current input token
             * NOTE: letting
             *   z[m][v+1] = logits[idx_m_vocab + ids_input[m+1])
             *   be the logit corresponding to the next input token, the
             *   expression below is equivalent to
             *
             *       -log(exp(z[m][v+1] - logits_m_max)) + log_sum_exp_m  =
             *   = -( log(exp(z[m][v+1] - logits_m_max)) - log_sum_exp_m) =
             *   = -log( (exp(z[m][v+1] - logits_m_max)) / sum_exp_m) ,
             *
             *  which implicitly applies softmax normalization to each logit to
             *  convert it into a probability */
            const auto next_input_id = ids_input[m+1];
            loss += -(logits[idx_m_vocab + next_input_id] - logits_m_max) + std::log(sum_exp_m);


            /* Normalize the softmax probabilities for each logit in the logits
             * vector for the current input token (i.e., for the current m
             * index) and accumulate the loss' gradients wrt the logits' weights
             * and biases and wrt the final inputs */
            const double sum_exp_m_inv = 1.0/sum_exp_m;

            for (auto v = decltype(nv){0}; v < nv; ++v) {
                probs[idx_m_vocab + v] *= sum_exp_m_inv;
            }

            probs[idx_m_vocab + next_input_id] -= 1.0;

            // Accumulate the loss' gradient wrt to the logits' weights
            for (auto v = decltype(nv){0}; v < nv; ++v) {
                const auto probs_mv = probs[idx_m_vocab + v];
                d_logits_b[v] += probs_mv;
            }
        }


        /* -------------------------------
         * 2. dL/d_logits_W = x3^T * probs
         * -------------------------------
         * - x3:         nx*DIM
         * - probs:      nx*nv
         * - d_logits_W: DIM*nv
         * NOTE: `x` holds the input tokens after LN3, so it really is `x3`
         * NOTE: `probs` should technically have dimensions (nx-1)*nv, but we
         *       allocate it as nx*nv and set its last row to zero to make use
         *       of cblas_dgemm */
        std::fill(probs.begin() + (nx-1)*nv, probs.end(), 0.0);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, nv, nx, 1.0, x.data(), DIM, probs.data(), nv,
                    0.0, d_logits_W.data(), nv);

        /* -----------------------------------
         * 3. dx = dL_dx3 = probs * logits_W^T
         * -----------------------------------
         * - probs:    nx*nv
         * - logits_W: DIM*nv
         * - dx = dx3: nx*DIM */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nx, DIM, nv, 1.0, probs.data(), nv, logits_W.data(), nv,
                    0.0, dx.data(), DIM);


        /* ---------------------------------------------------------------
         * 4. Backward LN3, dL/d_ffn_b2, and dL/dx2 [skip connection path]
         * --------------------------------------------------------------- */
        for (auto m = decltype(nx){0}; m < nx - 1; ++m) {
            const auto idx_m      = m*DIM;
            double u3_m_sum       = 0.0;
            double u3_xhat3_m_sum = 0.0;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi       = idx_m + i;
                const auto dx3_mi   = dx[mi];
                const auto xhat3_mi = xhat3[mi];

                d_shift3[i] += dx3_mi;
                d_scale3[i] += dx3_mi*xhat3_mi;

                /* This could be saved to a buffer and reused in the next loop (*),
                 * but a single multiplication is probably cheaper than a memory
                 * access */
                const auto u3_mi = dx3_mi*scale3[i];
                u3_m_sum        += u3_mi;
                u3_xhat3_m_sum  += u3_mi*xhat3_mi;
            }

            const auto sigma_inv3_m = sigmas_inv3[m];

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi    = idx_m + i;
                const auto u3_mi = dx[mi]*scale3[i];  // (*) HERE

                // Skip connection part of dL^m / dx2^m_i
                const auto dx2_skipconn_mi = sigma_inv3_m*(
                    u3_mi - (u3_m_sum + u3_xhat3_m_sum*xhat3[mi])*dim_inv);

                // NOTE: dx^m_i overwritten by dL^m / dx2skip^m_i
                dx[mi]       = dx2_skipconn_mi;
                d_ffn_b2[i] += dx2_skipconn_mi;
            }
        }


        /* --------------------------------------------
         * 5. dL/dFFN_W2 = d_ffn_W2 = ffn_h^T * dx2skip
         * --------------------------------------------
         * - ffn_h:        nx * dim_ffn_expanded  = nx * (FFN_EXPANSION_FACTOR * DIM)
         * - dx = dx2skip: nx * DIM
         * - d_ffn_W2:     dim_ffn_expanded * DIM = (FFN_EXPANSION_FACTOR * DIM) * DIM
         * NOTE: dx now holds dx2skip (overwritten in the loop above) */
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim_ffn_expanded, DIM, nx, 1.0, ffn_h.data(), dim_ffn_expanded, dx.data(), DIM,
                    0.0, d_ffn_W2.data(), DIM);


        /* ------------------------------------------------------
         * 6. dz^m_r = (dx2skip * ffn_W2^T)^m_r * ffn_h_prime^m_r
         * ------------------------------------------------------
         * - dx = dx2skip: nx * DIM
         * - ffn_W2:       dim_ffn_expanded * DIM = (FFN_EXPANSION_FACTOR * DIM) * DIM
         * - dz:           nx * dim_ffn_expanded  = nx * (FFN_EXPANSION_FACTOR * DIM)
         * NOTE: dx now holds dx2skip (overwritten in the loop above) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nx, dim_ffn_expanded, DIM, 1.0, dx.data(), DIM, ffn_W2.data(), DIM,
                    0.0, dz.data(), dim_ffn_expanded);

        for (auto idx = decltype(nxdim_expanded){0}; idx < nxdim_expanded; ++idx) {
            dz[idx] *= ffn_h_prime[idx];
        }


        /* ---------------------------
         * 7. Backward FFN first layer
         * --------------------------- */
        for (auto m = decltype(nx){0}; m < nx - 1; ++m) {
            const auto idx_m_exp = m*dim_ffn_expanded;
            for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
                d_ffn_b1[r] += dz[idx_m_exp + r];
            }
        }

        /* ***** dL/dFFN_W1 = d_ffn_W1 = x2^T * dz *****
         * - x2:       nx * DIM
         * - dz:       nx * dim_ffn_expanded  = nx * (FFN_EXPANSION_FACTOR * DIM)
         * - d_ffn_W1: DIM * dim_ffn_expanded */
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, dim_ffn_expanded, nx, 1.0, x2.data(), DIM, dz.data(), dim_ffn_expanded,
                    0.0, d_ffn_W1.data(), dim_ffn_expanded);


        /* ------------------------------------------------------------------
         * dL/dx2 [total] = dL/dx2 [skip connection path] + dL/dx2 [FFN path]
         * ------------------------------------------------------------------ */
        /* ***** dL/dx2 [FFN path] = dz * ffn_W1^T *****
         * - dz:             nx * dim_ffn_expanded = nx * (FFN_EXPANSION_FACTOR * DIM)
         * - ffn_W1:        DIM * dim_ffn_expanded
         * - dx = dx2 [FFN]: nx * DIM
         * NOTE: dx now holds dL/dx2 [total], note `1.0` before `dx.data()` */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nx, DIM, dim_ffn_expanded, 1.0, dz.data(), dim_ffn_expanded, ffn_W1.data(), dim_ffn_expanded,
                    1.0, dx.data(), DIM);  // NOTE: `1.0` adds dz * ffn_W1^T to the old dx (dL/dx2 [skip connection path]) 


        /* -------------------------------------------------
         * 9. Backward LN2 and dL/dx1 [skip connection path]
         * ------------------------------------------------- */
        // NOTE: dx now holds dL/dx2 [total]
        for (auto m = decltype(nx){0}; m < nx - 1; ++m) {
            const auto idx_m      = m*DIM;
            double u2_m_sum       = 0.0;
            double u2_xhat2_m_sum = 0.0;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi       = idx_m + i;
                const auto dx2_mi   = dx[mi];
                const auto xhat2_mi = xhat2[mi];

                d_shift2[i] += dx2_mi;
                d_scale2[i] += dx2_mi*xhat2_mi;

                /* This could be saved to a buffer and reused in the next loop (**),
                 * but a single multiplication is probably cheaper than a memory
                 * access */
                const auto u2_mi = dx2_mi*scale2[i];
                u2_m_sum        += u2_mi;
                u2_xhat2_m_sum  += u2_mi*xhat2_mi;
            }

            const auto sigma_inv2_m = sigmas_inv2[m];

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi    = idx_m + i;
                const auto u2_mi = dx[mi]*scale2[i];  // (**) HERE

                // Skip connection part of dL^m / dx1^m_i
                const auto dx1_skipconn_mi = sigma_inv2_m*(
                    u2_mi - (u2_m_sum + u2_xhat2_m_sum*xhat2[mi])*dim_inv);

                // NOTE: dx^m_i overwritten by dL^m / dx1skip^m_i
                dx[mi] = dx1_skipconn_mi;
            }
        }


        /* -----------------------------------------------
         * 10. Backward context
         *     a. dL/dA = dx1 [skip connection path] * V^T
         *     b. dL/dV = A^T * dx1 [skip connection path]
         * -----------------------------------------------
         * a. - dx1 [skip] = dx: nx * DIM
         *    - V = values:      nx * DIM
         *    - dL/dA = dA:      nx * nx
         * b. - A = attention:   nx * nx
         *    - dx1 [skip] = dx: nx * DIM
         *    - dL/dV = dV:      nx * DIM */
         cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                     nx, nx, DIM, 1.0, dx.data(), DIM, values.data(), DIM,
                     0.0, dA.data(), nx);
         cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                     nx, DIM, nx, 1.0, attention.data(), nx, dx.data(), DIM,
                     0.0, dV.data(), DIM);


        /* -----------------------------------------
         * 11. Backward attention softmax, per token
         *     dS = dL/dS = dL/dA * dA/dS
         * -----------------------------------------
         * dS[m,n] = / A[m,n]*(dA[m,n] - sum_{p<=m} A[m,p]*dA[m,p])  (n <= m)
         *           | 0                                             (n > m)
         * NOTE: dS overwrites dA */
         for (auto m = decltype(nx){0}; m < nx; ++m) {
             const auto    idx_m =  m*nx;
             const double *A_m   = &attention[idx_m];
                   double *dA_m  = &dA[idx_m];
                   double  dot   =  0.0;

             for (auto n = decltype(nx){0}; n <= m; ++n) {
                 dot += A_m[n]*dA_m[n];
             }

             for (auto n = decltype(nx){0}; n <= m; ++n) {
                 dA_m[n] = A_m[n]*(dA_m[n] - dot);
             }

             for (auto n = decltype(nx){m+1}; n < nx; ++n) {
                 dA_m[n] = 0.0;
             }
         }


        /* --------------------------------------------
         * 12. Backward query and key matrices
         *     a. dL/dQ = (1/sqrt(DIM)) *  dL/dS    * K
         *     b. dL/dK = (1/sqrt(DIM)) * (dL/dS)^T * Q
         * --------------------------------------------
         * a. - dL/dS = dS = dA:   nx * nx
         *    - K = keys:          nx * DIM
         *    - dL/dQ = dQ:        nx * DIM
         * b. - dL/dS = dS = dA:   nx * nx
         *    - Q = queries:       nx * DIM
         *    - dL/dK = dK:        nx * DIM
         * NOTE: dA now holds dS (see 11.) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nx, DIM, nx, sqrt_dim_inv, dA.data(), nx, keys.data(), DIM,
                    0.0, dQ.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nx, DIM, nx, sqrt_dim_inv, dA.data(), nx, queries.data(), DIM,
                    0.0, dK.data(), DIM);


        /* ------------------------------------------
         * 13. Backward query, key, and value weights
         *     a. dL/dWq = x1^T * dQ
         *     b. dL/dWk = x1^T * dK
         *     c. dL/dWv = x1^T * dV
         * ------------------------------------------
         * - x1:             nx * DIM
         * - dQ,  dK,  dV:   nx * DIM
         * - dWq, dWk, dWv: DIM * DIM */
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, DIM, nx, 1.0, x1.data(), DIM, dQ.data(), DIM,
                    0.0, dWq.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, DIM, nx, 1.0, x1.data(), DIM, dK.data(), DIM,
                    0.0, dWk.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, DIM, nx, 1.0, x1.data(), DIM, dV.data(), DIM,
                    0.0, dWv.data(), DIM);


        /* -----------------------------------------------------------------
         * 14. Backward x1
         *     dL/dx1 = dL/dx1 [skip] + dL/dx1 [Q] + dL/dx1 [K] + dL/dx1 [V]
         *     a. dL/dx1 [Q] = dQ * Wq^T
         *     b. dL/dx1 [K] = dK * Wk^T
         *     c. dL/dx1 [V] = dV * Wv^T
         * -----------------------------------------------------------------
         * NOTE: dx now holds dL/dx1 [skip connection path] ("dx1 [skip]") and
         *       the result (dx1 [total]) is accumulated back into dx
         * - dx1 [skip] = dx:  nx * DIM
         * - dQ, dK, dV:       nx * DIM
         * - Wq, Wk, Wv:      DIM * DIM */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nx, DIM, DIM, 1.0, dQ.data(), DIM, Wq.data(), DIM,
                    1.0, dx.data(), DIM);  // 1.0 adds dQ * Wq^T to dx1 [skip]
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nx, DIM, DIM, 1.0, dK.data(), DIM, Wk.data(), DIM,
                    1.0, dx.data(), DIM);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nx, DIM, DIM, 1.0, dV.data(), DIM, Wv.data(), DIM,
                    1.0, dx.data(), DIM);


        /* ---------------------------
         * 15. Backward LN1 and dL/dx0
         * --------------------------- */
        // NOTE: dx now holds dL/dx1 [total]
        for (auto m = decltype(nx){0}; m < nx - 1; ++m) {
            const auto idx_m = m*DIM;
            double u1_m_sum       = 0.0;
            double u1_xhat1_m_sum = 0.0;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi       = idx_m + i;
                const auto dx1_mi   = dx[mi];
                const auto xhat1_mi = xhat1[mi];

                d_shift1[i] += dx1_mi;
                d_scale1[i] += dx1_mi*xhat1_mi;

                /* This could be saved to a buffer and reused in the next loop (***),
                 * but a single multiplication is probably cheaper than a memory
                 * access */
                const auto u1_mi = dx1_mi*scale1[i];
                u1_m_sum        += u1_mi;
                u1_xhat1_m_sum  += u1_mi*xhat1_mi;
            }

            const auto sigma_inv1_m = sigmas_inv1[m];

            // ids_input[m] = v, i.e., the vocabulary's row index
            const auto idx_vocab = ids_input[m]*DIM;

            for (auto i = decltype(DIM){0}; i < DIM; ++i) {
                const auto mi    = idx_m + i;
                const auto u1_mi = dx[mi]*scale1[i];  // (***) HERE

                const auto dx0_mi = sigma_inv1_m*(
                    u1_mi - (u1_m_sum + u1_xhat1_m_sum*xhat1[mi])*dim_inv);

                /* Update the vocabulary and positional embedding vectors
                 * directly without extra buffers */
                vocab_embeddings[idx_vocab + i] -= params_norm_fac_sqrt_dim*dx0_mi;
                  pos_embeddings[mi]            -= params_norm_fac*dx0_mi;
            }

        }


        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX: update all the other model parameters
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX
        // XXX XXX XXX XXX XXX XXX


        /* ----------------------------------------------------------
         * Compute the average loss and update the model's parameters
         * ---------------------------------------------------------- */
        loss *= loss_norm_fac;
        loss_file << it << "\t" << loss << std::endl;

        for (auto v = decltype(nv){0}; v < nv; ++v) {
            logits_b[v] -= params_norm_fac*d_logits_b[v];
        }

        for (auto idx = decltype(nvdim){0}; idx < nvdim; ++idx) {
            logits_W[idx] -= params_norm_fac*d_logits_W[idx];
        }

        for (auto idx = decltype(dim_ffn_weights){0}; idx < dim_ffn_weights; ++idx) {
            ffn_W1[idx] -= params_norm_fac*d_ffn_W1[idx];
            ffn_W2[idx] -= params_norm_fac*d_ffn_W2[idx];
        }

        for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
            ffn_b1[r] -= params_norm_fac*d_ffn_b1[r];
        }

        for (auto i = decltype(DIM){0}; i < DIM; ++i) {
            ffn_b2[i] -= params_norm_fac*d_ffn_b2[i];

            scale1[i] -= params_norm_fac*d_scale1[i];
            scale2[i] -= params_norm_fac*d_scale2[i];
            scale3[i] -= params_norm_fac*d_scale3[i];

            shift1[i] -= params_norm_fac*d_shift1[i];
            shift2[i] -= params_norm_fac*d_shift2[i];
            shift3[i] -= params_norm_fac*d_shift3[i];
        }

        for (auto idx = decltype(dim_sq){0}; idx < dim_sq; ++idx) {
            Wq[idx] -= params_norm_fac*dWq[idx];
            Wk[idx] -= params_norm_fac*dWk[idx];
            Wv[idx] -= params_norm_fac*dWv[idx];
        }

        if constexpr (VERBOSE) {
            std::cout << "Training epoch " << it << " completed" << std::endl;
        }

        //// XXX
        //// Predict the next token
        //const auto last_token_idx = nx*nv - nv;  // (nx-1)*nv
        //std::vector<double> last_logits(nv);
        //for (auto v = decltype(nv){0}; v < nv; ++v) {
        //    last_logits[v] = logits[last_token_idx + v];
        //}
        //const auto new_id    = distance(last_logits.begin(), max_element(last_logits.begin(), last_logits.end()));
        //const auto new_token = tokenizer.decode(std::vector<size_t>{new_id});
        //std::cout << input_text << " " << new_token << std::endl;
        //// XXX
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
    //        auto &io = ios[n];
    //        output = tokens[n + CONTEXT_SIZE];

    //        for (auto i = decltype(CONTEXT_SIZE){0}; i < CONTEXT_SIZE; ++i) {
    //            io[i] = tokens[n + i];
    //        }

    //        io[CONTEXT_SIZE] = 
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
