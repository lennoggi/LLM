/* ========================================
 * Model (transformer) class implementation
 * ======================================== */
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

#include <cblas.h>

#include "functions.hh"
#include "types.hh"
#include "parameters.hh"


/* ===========
 * Constructor
 * =========== */
transformer_t::transformer_t(
    const size_t &nv,
    std::mt19937 &gen) : nv(nv), gen(gen), udist(0.0, 1.0)
{
    /* -------------------------
     * Constants used repeatedly
     * ------------------------- */
    constexpr auto cw               = CONTEXT_WINDOW;  // Just a handy alias
    constexpr auto cwsq             = cw*cw;
    constexpr auto dim_sq           = DIM*DIM;
    constexpr auto dim_ffn_expanded = DIM*FFN_EXPANSION_FACTOR;
    constexpr auto dim_ffn_weights  = DIM*dim_ffn_expanded;
    constexpr auto cwdim            = cw*DIM;
    constexpr auto cwdim_expanded   = cw*dim_ffn_expanded;
    const     auto nvdim            = nv*DIM;
    const     auto cwnv             = cw*nv;


    /* -------------------------------
     * Initialize learnable parameters
     * ------------------------------- */
    // Model parameters count
    const auto nparams =
        nvdim                      +  // vocab_embeddings
        cwdim                      +  // pos_embeddings
        6*DIM                      +  // LN scale1/2/3, shift1/2/3
        3*dim_sq                   +  // Wq, Wk, Wv
        2*dim_ffn_weights          +  // ffn_W1, ffn_W2
        dim_ffn_expanded + DIM     +  // ffn_b1, ffn_b2
        (WEIGHT_TYING ? 0 : nvdim) +  // logits_W
        nv;                           // logits_b

    std::cout << "INFO: the model has " << nparams << " parameters" << std::endl;


    /* Initialize a vector representation ("embedding") of each token in the
     * vocabulary with Gaussian random values to be optimized during training */
    std::normal_distribution<double> ndist(0.0, 1.0/std::sqrt(static_cast<double>(DIM)));

    vocab_embeddings.resize(nvdim);  // Matrix (nv, DIM)
      pos_embeddings.resize(cwdim);  // Matrix (cw, DIM)

    for (auto &el : vocab_embeddings) {
        el = ndist(gen);
    }

    for (auto &el : pos_embeddings) {
        el = ndist(gen);
    }


    /* Initialize the three layer normalization (LN) scale and shift vectors to
     * 1's and 0's, respectively */
    // 1. LN before the attention block
    scale1.assign(DIM, 1.0);
    shift1.assign(DIM, 0.0);

    // 2. LN before the feed-forward neural network (FFN)
    scale2.assign(DIM, 1.0);
    shift2.assign(DIM, 0.0);

    // 3. LN before predicting the new token
    scale3.assign(DIM, 1.0);
    shift3.assign(DIM, 0.0);


    /* Initialize the query, key, and value weight matrices with random values
     * drawn from a Xavier/Glorot uniform distribution */
    constexpr auto xg_dim_bound = std::sqrt(3.0/(static_cast<double>(DIM)));
    std::uniform_real_distribution<double> xg_dim_udist(-xg_dim_bound, xg_dim_bound);

    Wq.resize(dim_sq);  // Matrix (DIM, DIM)
    Wk.resize(dim_sq);  // Matrix (DIM, DIM)
    Wv.resize(dim_sq);  // Matrix (DIM, DIM)

    for (auto idx = decltype(dim_sq){0}; idx < dim_sq; ++idx) {
        Wq[idx] = xg_dim_udist(gen);
        Wk[idx] = xg_dim_udist(gen);
        Wv[idx] = xg_dim_udist(gen);
    }


    /* Initialize the feed-forward neural network (FFN) weights with random
     * values drawn from a Xavier/Glorot normal distribution, and set the biases
     * to zero */
    constexpr auto xg_ffn_std = std::sqrt(2.0/(static_cast<double>(DIM) + static_cast<double>(dim_ffn_expanded)));
    std::normal_distribution<double> xg_ffn_ndist(0.0, xg_ffn_std);

    ffn_W1.resize(dim_ffn_weights);  // Matrix (DIM, DIM*FFN_EXPANSION_FACTOR)
    ffn_W2.resize(dim_ffn_weights);  // Matrix (DIM*FFN_EXPANSION_FACTOR, DIM)

    for (auto idx = decltype(dim_ffn_weights){0}; idx < dim_ffn_weights; ++idx) {
        ffn_W1[idx] = xg_ffn_ndist(gen);
        ffn_W2[idx] = xg_ffn_ndist(gen);
    }

    ffn_b1.assign(dim_ffn_expanded, 0.0);
    ffn_b2.assign(DIM, 0.0);


    /* If weight tying is active, use the vocabulary embedding as the logits
     * weights; otherwise, initialize the logits weights to random values.
     * Regardless, set the logits biases to zero. */
    if constexpr (not WEIGHT_TYING) {
        logits_W.resize(nvdim);  // Matrix (DIM, nv)
        for (auto &el : logits_W) {
            el = ndist(gen);
        }
    }

    logits_b.assign(nv, 0.0);


    /* ----------------------
     * Allocate the gradients
     * ---------------------- */
    d_vocab_embeddings.resize(nvdim);
      d_pos_embeddings.resize(cwdim);

    d_scale1.resize(DIM);
    d_shift1.resize(DIM);

    d_scale2.resize(DIM);
    d_shift2.resize(DIM);

    d_scale3.resize(DIM);
    d_shift3.resize(DIM);

    dWq.resize(dim_sq);
    dWk.resize(dim_sq);
    dWv.resize(dim_sq);

    d_ffn_W1.resize(dim_ffn_weights);
    d_ffn_W2.resize(dim_ffn_weights);

    d_ffn_b1.resize(dim_ffn_expanded);
    d_ffn_b2.resize(DIM);

    d_logits_b.resize(nv);
    d_logits_W.resize(nvdim);


    /* ------------------------------
     * Allocate the forward workspace
     * ------------------------------ */
     x.resize(cwdim);
    x1.resize(cwdim);
    x2.resize(cwdim);

    queries.resize(cwdim);
       keys.resize(cwdim);
     values.resize(cwdim);

    attention.resize(cwsq);
      context.resize(cwdim);

    xhat1.resize(cwdim);
    xhat2.resize(cwdim);
    xhat3.resize(cwdim);

    sigmas_inv1.resize(cw);
    sigmas_inv2.resize(cw);
    sigmas_inv3.resize(cw);

          ffn_h.resize(cwdim_expanded);
    ffn_h_prime.resize(cwdim_expanded);
        ffn_out.resize(cwdim);

    logits.resize(cwnv);


    /* -------------------------------
     * Allocate the backward workspace
     * ------------------------------- */
    probs.resize(cwnv);
    dx.resize(cwdim);
    dz.resize(cwdim_expanded);
    dA.resize(cwsq);

    dQ.resize(cwdim);
    dK.resize(cwdim);
    dV.resize(cwdim);
}



/* ========================================
 * Forward step: turn token IDs into logits
 * ======================================== */
void transformer_t::forward(
    const std::vector<size_t> &ids,
    const size_t &start_id_train)
{
    constexpr double sqrt_2_over_pi = std::sqrt(2.0/M_PI);
    constexpr double gelu_a         = 0.044715;

    constexpr double sqrt_dim     = std::sqrt(static_cast<double>(DIM));
    constexpr double sqrt_dim_inv = 1.0/sqrt_dim;

    constexpr auto cw               = CONTEXT_WINDOW;  // Just a handy alias
    constexpr auto dim_ffn_expanded = DIM*FFN_EXPANSION_FACTOR;
    constexpr auto cwdim_expanded   = cw*dim_ffn_expanded;

    const auto *ids_train_shifted = &ids[start_id_train];


    /* --------------------------
     * Build the vector embedding
     * --------------------------
     * Map each training token ID withing the current window into the
     * corresponding embedding vector (scaled by sqrt(DIM) to keep
     * magnitudes consistent) and add the corresponding positional
     * encoding vectors */
    for (auto m = decltype(cw){0}; m < cw; ++m) {
        const auto idx_m    = m*DIM;
        const auto token_id = ids_train_shifted[m];
        assert(token_id < nv);
        const auto idx_vocab = token_id*DIM;

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
     * - x:          cw*DIM
     * - Wq, Wk, Wv: DIM*DIM
     * - queries, keys, values: cw*DIM */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cw, DIM, DIM, 1.0, x.data(), DIM, Wq.data(), DIM,
        0.0, queries.data(), DIM);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cw, DIM, DIM, 1.0, x.data(), DIM, Wk.data(), DIM,
        0.0, keys.data(), DIM);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cw, DIM, DIM, 1.0, x.data(), DIM, Wv.data(), DIM,
        0.0, values.data(), DIM);

    /* --------------------------------------------
     * attention = (1/sqrt(DIM)) * queries * keys^T
     * --------------------------------------------
     * - queries:   cw*DIM
     * - keys^T:    DIM*cw
     * - attention: cw*cw
     * TODO: allow for multi-head attention; need to swap
     *   nds_input<->nheads to allow parallelization by head. Then the
     *   normalization factor will become 1/sqrt(DIM_OUT/nheads) */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        cw, cw, DIM, sqrt_dim_inv, queries.data(), DIM, keys.data(), DIM,
        0.0, attention.data(), cw);


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
    for (auto m = decltype(cw){0}; m < cw; ++m) {
              double *attention_m = &attention[m*cw];
        /* Restrict the maximum to the causal (i.e., lower-diagonal) part
         * of the attention matrix                                          */
        const double  attention_m_max = *std::max_element(attention_m, attention_m + m + 1);
              double  sum_exp = 0.0;

        for (auto n = decltype(cw){0}; n <= m; ++n) {
            attention_m[n] = std::exp(attention_m[n] - attention_m_max);
            sum_exp       += attention_m[n];
        }

        const double inv_sum_exp = 1.0/sum_exp;

        for (auto n = decltype(cw){0}; n <= m; ++n) {
            attention_m[n] *= inv_sum_exp;
        }

        for (auto n = decltype(cw){m+1}; n < cw; ++n) {
            attention_m[n] = 0.0;
        }
    }


    /* ----------------------------
     * context = attention * values
     * ----------------------------
     * - attention: cw*cw
     * - values:    cw*DIM
     * - context    cw*DIM */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cw, DIM, cw, 1.0, attention.data(), cw, values.data(), DIM,
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
     *                            |   ...    | | dim_ffn_expanded times
     *                            \ [ffn_b1] / |
     * - x:      cw*DIM
     * - ffn_W1: DIM*(DIM*FFN_EXPANSION_FACTOR) = DIM*dim_ffn_expanded
     * - ffn_h:  cw*dim_ffn_expanded */
    for (auto m = decltype(cw){0}; m < cw; ++m) {
        std::copy(ffn_b1.begin(), ffn_b1.end(), ffn_h.begin() + m*dim_ffn_expanded);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cw, dim_ffn_expanded, DIM, 1.0, x.data(), DIM, ffn_W1.data(), dim_ffn_expanded,
        1.0, ffn_h.data(), dim_ffn_expanded);

    /* STEP 2: take (an approximation of) GELU(ffn_h)
     * NOTE: derivative of pre-GELU inputs needed for the backward pass */
    for (auto idx = decltype(cwdim_expanded){0}; idx < cwdim_expanded; ++idx) {
        auto &z = ffn_h[idx];
        const auto az2 = gelu_a*z*z;
        const auto sqrt_2_over_pi_z = sqrt_2_over_pi*z;
        const auto th   = tanh(sqrt_2_over_pi_z*(1.0 + az2));
        const auto thp1 = th + 1.0;

        z *= 0.5*thp1;
        ffn_h_prime[idx] = 0.5*(thp1 + sqrt_2_over_pi_z*(1.0 + 3.0*az2)*(1.0 - th*th));
    }

    /* STEP 3: ffn_out = ffn_h*ffn_W2 + / [ffn_b2] \ |
     *                                  |   ...    | | DIM times
     *                                  \ [ffn_b2] / |
     * - ffn_h:  cw*dim_ffn_expanded
     * - ffn_W2: (DIM*FFN_EXPANSION_FACTOR)*DIM = dim_ffn_expanded*DIM
     * - ffn_out: cw*DIM */
    for (auto m = decltype(cw){0}; m < cw; ++m) {
        std::copy(ffn_b2.begin(), ffn_b2.end(), ffn_out.begin() + m*DIM);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cw, DIM, dim_ffn_expanded, 1.0, ffn_h.data(), dim_ffn_expanded, ffn_W2.data(), DIM,
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


    /* -------------------------------------------------------------
     * logits = x*logits_W + / [logits_b] \ |
     *                       | [logits_b] | | nv times
     *                       \ [logits_b] / |
     * NOTE: logits_W = vocab_embeddings^T if weight tying is active
     * -------------------------------------------------------------
     * - x:        cw*DIM
     * - logits_W: DIM*nv
     * - logits:   cw*nv */
    for (auto m = decltype(cw){0}; m < cw; ++m) {
        std::copy(logits_b.begin(), logits_b.end(), logits.begin() + m*nv);
    }

    if constexpr (WEIGHT_TYING) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            cw, nv, DIM, 1.0, x.data(), DIM, vocab_embeddings.data(), DIM,
            1.0, logits.data(), nv);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            cw, nv, DIM, 1.0, x.data(), DIM, logits_W.data(), nv,
            1.0, logits.data(), nv);
    }


    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    /* XXX: softmax is not really needed now because it doesn't change which
     *   logit is the largest (and so it doesn't change the next predicted
     *   token). However, we'll later introduce some variability, whereby
     *   the next predicted token is not the one corresponding to the
     *   largest logit, and softmax will be useful then. */
    //softmax(last_logits);  // XXX: last_logits defined below
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX

    return;
}



/* ===========================================================================
 * Backward pass: minimize the sum of the cross-entropy losses between each
 * training token and the next one via stochastic gradient descent and update
 * the model's parameters. This routine is intended to be called from within a
 * loop over all training sequences for a given training epoch and returns the
 * loss over the current sequence.
 * =========================================================================== */
double transformer_t::backward(
    const std::vector<size_t> &ids,
    const size_t &start_id_train)
{
    constexpr double dim_inv = 1.0/static_cast<double>(DIM);

    constexpr auto dim_sq           = DIM*DIM;
    constexpr auto dim_ffn_expanded = DIM*FFN_EXPANSION_FACTOR;
    constexpr auto dim_ffn_weights  = DIM*dim_ffn_expanded;

    constexpr auto cw               = CONTEXT_WINDOW;  // Just a handy alias
    constexpr auto cwdim_expanded   = cw*dim_ffn_expanded;
    const     auto nvdim            = nv*DIM;

    const auto *ids_train_shifted = &ids[start_id_train];

    double sequence_loss = 0.0;

    std::fill(d_ffn_b1.begin(), d_ffn_b1.end(), 0.0);
    std::fill(d_ffn_b2.begin(), d_ffn_b2.end(), 0.0);

    std::fill(d_scale1.begin(), d_scale1.end(), 0.0);
    std::fill(d_scale2.begin(), d_scale2.end(), 0.0);
    std::fill(d_scale3.begin(), d_scale3.end(), 0.0);

    std::fill(d_shift1.begin(), d_shift1.end(), 0.0);
    std::fill(d_shift2.begin(), d_shift2.end(), 0.0);
    std::fill(d_shift3.begin(), d_shift3.end(), 0.0);

    std::fill(d_logits_b.begin(), d_logits_b.end(), 0.0);


    /* --------------------------------------------------------------
     * 1. Compute the loss function for the current training sequence
     *    and build
     *      probs^m_v = dL^m / dlogits^m_v =
     *    = softmax(logits)^m_v - delta_{v, v_next_input_token}
     * -------------------------------------------------------------- */
    for (auto m = decltype(cw){0}; m < cw - 1; ++m) {
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

        /* Add the loss term for the current training token
         * NOTE: letting
         *   z[m][v+1] = logits[idx_m_vocab + ids_train_shifted[m+1])
         *   be the logit corresponding to the next input token, the
         *   expression below is equivalent to
         *
         *       -log(exp(z[m][v+1] - logits_m_max)) + log_sum_exp_m  =
         *   = -( log(exp(z[m][v+1] - logits_m_max)) - log_sum_exp_m) =
         *   = -log( (exp(z[m][v+1] - logits_m_max)) / sum_exp_m) ,
         *
         *  which implicitly applies softmax normalization to each logit to
         *  convert it into a probability */
        const auto next_token_id = ids_train_shifted[m+1];
        sequence_loss += -(logits[idx_m_vocab + next_token_id] - logits_m_max) + std::log(sum_exp_m);


        /* Normalize the softmax probabilities for each logit in the logits
         * vector for the current input token (i.e., for the current m
         * index) and accumulate the loss' gradients wrt the logits' weights
         * and biases and wrt the final inputs */
        const double sum_exp_m_inv = 1.0/sum_exp_m;

        for (auto v = decltype(nv){0}; v < nv; ++v) {
            probs[idx_m_vocab + v] *= sum_exp_m_inv;
        }

        probs[idx_m_vocab + next_token_id] -= 1.0;

        // Accumulate the loss' gradient wrt to the logits' weights
        for (auto v = decltype(nv){0}; v < nv; ++v) {
            const auto probs_mv = probs[idx_m_vocab + v];
            d_logits_b[v] += probs_mv;
        }
    }


    /* -------------------------------------------------------------
     * 2. Without weight tying: dL/d_logits_W         = x3^T * probs
     *    With    weight tying: dL/d_vocab_embeddings = probs^T * x3
     * -------------------------------------------------------------
     * - x3:         cw*DIM
     * - probs:      cw*nv
     * - d_logits_W: DIM*nv without weight tying, nv*DIM otherwise
     * NOTE: `x` holds the input tokens after LN3, so it really is `x3`
     * NOTE: `probs` should technically have dimensions (cw-1)*nv, but we
     *       allocate it as cw*nv and set its last row to zero to make use
     *       of cblas_dgemm
     * NOTE: d_logits_W is really "d_vocab_embeddings" if weight tying is
     *       active */
    std::fill(probs.begin() + (cw-1)*nv, probs.end(), 0.0);

    if constexpr (WEIGHT_TYING) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nv, DIM, cw, 1.0, probs.data(), nv, x.data(), DIM,
                    0.0, d_logits_W.data(), DIM);
    } else {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, nv, cw, 1.0, x.data(), DIM, probs.data(), nv,
                    0.0, d_logits_W.data(), nv);
    }


    /* -----------------------------------
     * 3. dx = dL_dx3 = probs * logits_W^T
     * -----------------------------------
     * - probs:    cw*nv
     * - logits_W: DIM*nv
     * - dx = dx3: cw*DIM
     * NOTE: logits_W = vocab_embeddings^T if weight tying is active */
    if constexpr (WEIGHT_TYING) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    cw, DIM, nv, 1.0, probs.data(), nv, vocab_embeddings.data(), DIM,
                    0.0, dx.data(), DIM);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    cw, DIM, nv, 1.0, probs.data(), nv, logits_W.data(), nv,
                    0.0, dx.data(), DIM);
    }


    /* ---------------------------------------------------------------
     * 4. Backward LN3, dL/d_ffn_b2, and dL/dx2 [skip connection path]
     * --------------------------------------------------------------- */
    for (auto m = decltype(cw){0}; m < cw - 1; ++m) {
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
     * - ffn_h:        cw * dim_ffn_expanded  = cw * (FFN_EXPANSION_FACTOR * DIM)
     * - dx = dx2skip: cw * DIM
     * - d_ffn_W2:     dim_ffn_expanded * DIM = (FFN_EXPANSION_FACTOR * DIM) * DIM
     * NOTE: dx now holds dx2skip (overwritten in the loop above) */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                dim_ffn_expanded, DIM, cw, 1.0, ffn_h.data(), dim_ffn_expanded, dx.data(), DIM,
                0.0, d_ffn_W2.data(), DIM);


    /* ------------------------------------------------------
     * 6. dz^m_r = (dx2skip * ffn_W2^T)^m_r * ffn_h_prime^m_r
     * ------------------------------------------------------
     * - dx = dx2skip: cw * DIM
     * - ffn_W2:       dim_ffn_expanded * DIM = (FFN_EXPANSION_FACTOR * DIM) * DIM
     * - dz:           cw * dim_ffn_expanded  = cw * (FFN_EXPANSION_FACTOR * DIM)
     * NOTE: dx now holds dx2skip (overwritten in the loop above) */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                cw, dim_ffn_expanded, DIM, 1.0, dx.data(), DIM, ffn_W2.data(), DIM,
                0.0, dz.data(), dim_ffn_expanded);

    for (auto idx = decltype(cwdim_expanded){0}; idx < cwdim_expanded; ++idx) {
        dz[idx] *= ffn_h_prime[idx];
    }


    /* ---------------------------
     * 7. Backward FFN first layer
     * --------------------------- */
    for (auto m = decltype(cw){0}; m < cw - 1; ++m) {
        const auto idx_m_exp = m*dim_ffn_expanded;
        for (auto r = decltype(dim_ffn_expanded){0}; r < dim_ffn_expanded; ++r) {
            d_ffn_b1[r] += dz[idx_m_exp + r];
        }
    }

    /* ***** dL/dFFN_W1 = d_ffn_W1 = x2^T * dz *****
     * - x2:       cw * DIM
     * - dz:       cw * dim_ffn_expanded  = cw * (FFN_EXPANSION_FACTOR * DIM)
     * - d_ffn_W1: DIM * dim_ffn_expanded */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                DIM, dim_ffn_expanded, cw, 1.0, x2.data(), DIM, dz.data(), dim_ffn_expanded,
                0.0, d_ffn_W1.data(), dim_ffn_expanded);


    /* ------------------------------------------------------------------
     * dL/dx2 [total] = dL/dx2 [skip connection path] + dL/dx2 [FFN path]
     * ------------------------------------------------------------------ */
    /* ***** dL/dx2 [FFN path] = dz * ffn_W1^T *****
     * - dz:             cw * dim_ffn_expanded = cw * (FFN_EXPANSION_FACTOR * DIM)
     * - ffn_W1:        DIM * dim_ffn_expanded
     * - dx = dx2 [FFN]: cw * DIM
     * NOTE: dx now holds dL/dx2 [total], note `1.0` before `dx.data()` */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                cw, DIM, dim_ffn_expanded, 1.0, dz.data(), dim_ffn_expanded, ffn_W1.data(), dim_ffn_expanded,
                1.0, dx.data(), DIM);  // NOTE: `1.0` adds dz * ffn_W1^T to the old dx (dL/dx2 [skip connection path])


    /* -------------------------------------------------
     * 9. Backward LN2 and dL/dx1 [skip connection path]
     * ------------------------------------------------- */
    // NOTE: dx now holds dL/dx2 [total]
    for (auto m = decltype(cw){0}; m < cw - 1; ++m) {
        const auto idx_m      = m*DIM;
        double u2_m_sum       = 0.0;
        double u2_xhat2_m_sum = 0.0;

        for (auto i = decltype(DIM){0}; i < DIM; ++i) {
            const auto mi       = idx_m + i;
            const auto dx2_mi   = dx[mi];
            const auto xhat2_mi = xhat2[mi];

            d_shift2[i] += dx2_mi;
            d_scale2[i] += dx2_mi*xhat2_mi;

            /* This could be saved to a buffer and reused in the next
             * loop (**), but a single multiplication is probably
             * cheaper than a memory access */
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
     * a. - dx1 [skip] = dx: cw * DIM
     *    - V = values:      cw * DIM
     *    - dL/dA = dA:      cw * cw
     * b. - A = attention:   cw * cw
     *    - dx1 [skip] = dx: cw * DIM
     *    - dL/dV = dV:      cw * DIM */
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                 cw, cw, DIM, 1.0, dx.data(), DIM, values.data(), DIM,
                 0.0, dA.data(), cw);
     cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                 cw, DIM, cw, 1.0, attention.data(), cw, dx.data(), DIM,
                 0.0, dV.data(), DIM);


    /* -----------------------------------------
     * 11. Backward attention softmax, per token
     *     dS = dL/dS = dL/dA * dA/dS
     * -----------------------------------------
     * dS[m,n] = / A[m,n]*(dA[m,n] - sum_{p<=m} A[m,p]*dA[m,p])  (n <= m)
     *           | 0                                             (n > m)
     * NOTE: dS overwrites dA */
     for (auto m = decltype(cw){0}; m < cw; ++m) {
         const auto    idx_m =  m*cw;
         const double *A_m   = &attention[idx_m];
               double *dA_m  = &dA[idx_m];
               double  dot   =  0.0;

         for (auto n = decltype(cw){0}; n <= m; ++n) {
             dot += A_m[n]*dA_m[n];
         }

         for (auto n = decltype(cw){0}; n <= m; ++n) {
             dA_m[n] = A_m[n]*(dA_m[n] - dot);
         }

         for (auto n = decltype(cw){m+1}; n < cw; ++n) {
             dA_m[n] = 0.0;
         }
     }


    /* --------------------------------------------
     * 12. Backward query and key matrices
     *     a. dL/dQ = (1/sqrt(DIM)) *  dL/dS    * K
     *     b. dL/dK = (1/sqrt(DIM)) * (dL/dS)^T * Q
     * --------------------------------------------
     * a. - dL/dS = dS = dA:   cw * cw
     *    - K = keys:          cw * DIM
     *    - dL/dQ = dQ:        cw * DIM
     * b. - dL/dS = dS = dA:   cw * cw
     *    - Q = queries:       cw * DIM
     *    - dL/dK = dK:        cw * DIM
     * NOTE: dA now holds dS (see 11.) */
    constexpr double sqrt_dim     = std::sqrt(static_cast<double>(DIM));
    constexpr double sqrt_dim_inv = 1.0/sqrt_dim;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                cw, DIM, cw, sqrt_dim_inv, dA.data(), cw, keys.data(), DIM,
                0.0, dQ.data(), DIM);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                cw, DIM, cw, sqrt_dim_inv, dA.data(), cw, queries.data(), DIM,
                0.0, dK.data(), DIM);


    /* ------------------------------------------
     * 13. Backward query, key, and value weights
     *     a. dL/dWq = x1^T * dQ
     *     b. dL/dWk = x1^T * dK
     *     c. dL/dWv = x1^T * dV
     * ------------------------------------------
     * - x1:             cw * DIM
     * - dQ,  dK,  dV:   cw * DIM
     * - dWq, dWk, dWv: DIM * DIM */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                DIM, DIM, cw, 1.0, x1.data(), DIM, dQ.data(), DIM,
                0.0, dWq.data(), DIM);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                DIM, DIM, cw, 1.0, x1.data(), DIM, dK.data(), DIM,
                0.0, dWk.data(), DIM);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                DIM, DIM, cw, 1.0, x1.data(), DIM, dV.data(), DIM,
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
     * - dx1 [skip] = dx:  cw * DIM
     * - dQ, dK, dV:       cw * DIM
     * - Wq, Wk, Wv:      DIM * DIM */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                cw, DIM, DIM, 1.0, dQ.data(), DIM, Wq.data(), DIM,
                1.0, dx.data(), DIM);  // 1.0 adds dQ * Wq^T to dx1 [skip]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                cw, DIM, DIM, 1.0, dK.data(), DIM, Wk.data(), DIM,
                1.0, dx.data(), DIM);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                cw, DIM, DIM, 1.0, dV.data(), DIM, Wv.data(), DIM,
                1.0, dx.data(), DIM);


    /* ---------------------------
     * 15. Backward LN1 and dL/dx0
     * --------------------------- */
    // NOTE: dx now holds dL/dx1 [total]
    constexpr double params_norm_fac = LEARNING_RATE/static_cast<double>(cw-1);
    constexpr double params_norm_fac_sqrt_dim = params_norm_fac*sqrt_dim;

    for (auto m = decltype(cw){0}; m < cw - 1; ++m) {
        const auto idx_m = m*DIM;
        double u1_m_sum       = 0.0;
        double u1_xhat1_m_sum = 0.0;

        for (auto i = decltype(DIM){0}; i < DIM; ++i) {
            const auto mi       = idx_m + i;
            const auto dx1_mi   = dx[mi];
            const auto xhat1_mi = xhat1[mi];

            d_shift1[i] += dx1_mi;
            d_scale1[i] += dx1_mi*xhat1_mi;

            /* This could be saved to a buffer and reused in the next
             * loop (***), but a single multiplication is probably
             * cheaper than a memory access */
            const auto u1_mi = dx1_mi*scale1[i];
            u1_m_sum        += u1_mi;
            u1_xhat1_m_sum  += u1_mi*xhat1_mi;
        }

        const auto sigma_inv1_m = sigmas_inv1[m];

        // ids_train_shifted[m] = v, i.e., the vocabulary's row index
        const auto idx_vocab = ids_train_shifted[m]*DIM;

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


    /* -----------------------------
     * Update the model's parameters
     * ----------------------------- */
    // XXX: update epoch_loss in main
    //epoch_loss += sequence_loss;

    for (auto v = decltype(nv){0}; v < nv; ++v) {
        logits_b[v] -= params_norm_fac*d_logits_b[v];
    }

    if constexpr (WEIGHT_TYING) {
        for (auto idx = decltype(nvdim){0}; idx < nvdim; ++idx) {
            vocab_embeddings[idx] -= params_norm_fac*d_logits_W[idx];
        }
    } else {
        for (auto idx = decltype(nvdim){0}; idx < nvdim; ++idx) {
            logits_W[idx] -= params_norm_fac*d_logits_W[idx];
        }
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

    return sequence_loss;
}
