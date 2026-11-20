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
#include "functions.hh"
#include "parameters.hh"


int main() {
    /* ------------------------
     * Encode the training text
     * ------------------------ */
    const auto training_text = file_to_string(TRAINING_FILE);

    // TODO: use this as the prompt
    //const auto input_text = file_to_string(INPUT_FILE);

    #if (TOKENIZER == WORD)
    auto tokenizer = word_tokenizer_t(training_text);
    #elif (TOKENIZER == BPE)
    auto tokenizer = bpe_tokenizer_t(training_text, BPE_END_OF_WORD, BPE_MAX_VOCAB_SIZE);
    #else
    #error "ERROR: Invalid tokenizer"
    return 1;  // Not reached
    #endif

    /* Reserve a fraction VALIDATION_FRACTION of the text to validating the
     * training */
    const auto &ids_all   = tokenizer.encode(training_text);
    const auto nids_all   = ids_all.size();
    const auto nids_val   = static_cast<decltype(nids_all)>(VALIDATION_FRACTION*static_cast<double>(nids_all));
    const auto nids_train = nids_all - nids_val;

    constexpr auto cw = CONTEXT_WINDOW;  // Just a handy alias

    if (nids_train < cw or nids_val < cw) {
        throw std::runtime_error("The training or validation part of the training text is shorter than CONTEXT_WINDOW");
        return 1;  // Not reached
    }

    const auto nv = tokenizer.vocab_token2id.size();


    /* ------------------------------------
     * Initialize a random number generator
     * ------------------------------------ */
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


    /* -----------------------------------------------------------
     * Open a file to log the loss function at each training epoch
     * ----------------------------------------------------------- */
    std::ofstream loss_file(LOSS_FILE);

    if (not loss_file.is_open()) {
        std::ostringstream err_ss;
        err_ss << "Failed to write to file '" << LOSS_FILE << "'";
        throw std::runtime_error(err_ss.str());
        return 1;  // Not reached
    }

    loss_file << "# Column 1: training epoch" << std::endl
              << "# Column 2: loss function averaged over all training or validation tokens and all context windows" << std::endl;


    /* ----------------------------------
     *  Print OpenBLAS configuration info
     * ---------------------------------- */
    std::cout << "INFO: OpenBLAS settings" << std::endl;
    const auto openblas_parallel_mode = openblas_get_parallel();

    if (openblas_parallel_mode == 0) {
        std::cout << "      - OpenBLAS parallel mode: single-threaded" << std::endl;
    } else if (openblas_parallel_mode == 1) {
        std::cout << "      - OpenBLAS parallel mode: pthreads or native OS threads" << std::endl;
    } else if (openblas_parallel_mode == 2) {
        std::cout << "      - OpenBLAS parallel mode: OpenMP threads" << std::endl;
    } else {
        std::cout << "ERROR: unexpected OpenBLAS parallel mode ("
                  << openblas_parallel_mode << ")" << std::endl;
    }   

    std::cout << "      - OpenBLAS configuration: " << openblas_get_config()      << std::endl;
    std::cout << "      - Core architecture:      " << openblas_get_corename()    << std::endl;
    std::cout << "      - Number of threads:      " << openblas_get_num_threads() << std::endl;


    /* ========
     * Training
     * ======== */
    transformer_t model(nv, gen);

    /* Build a vector storing the start ids of all the training sequences,
     * where each "regular" sequence is cw tokens long. If the last regular
     * window stops short of the end of the training text, add one full-CW
     * window aligned with the end of the text. Note that this effectively sets
     * a smaller stride for the last sequence: not a big deal.
     * EXAMPLE: N = 20, cw = 6, stride = 6/2 = 3
     *   Training IDs:  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
     *   Indices:       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
     *   Start indices: +        +        +        +        +     *
     *                  0        1        2        3       *4*    5: extra sequence
     *   Last full-cw sequence index: (N - cw)/stride = (20-6)/3 = 4
     *   Number of full-CW sequences = 4 + 1 = 5  // nwindows_fullcw */
    const size_t nwindows_fullcw = (nids_train - cw)/CONTEXT_STRIDE + 1;
    std::vector<size_t> start_ids_train(nwindows_fullcw);

    for (size_t w = 0; w < nwindows_fullcw; ++w) {
        start_ids_train[w] = w*CONTEXT_STRIDE;
    }

    if (start_ids_train.back() + cw < nids_train) {
        start_ids_train.push_back(nids_train - cw);
    }


    /* -------------
     * Training loop
     * ------------- */
    const double loss_norm_fac_epoch = 1.0/static_cast<double>((cw-1)*start_ids_train.size());  // Only used to log the loss
    const auto   training_start      = std::chrono::steady_clock::now();

    for (auto epoch = decltype(NEPOCHS){0}; epoch < NEPOCHS; ++epoch) {
        /* Randomly shuffle the training sequences to improve the generalization
         * capabilities of the model */
        std::shuffle(start_ids_train.begin(), start_ids_train.end(), gen);

        double epoch_loss = 0.0;

        for (const auto &start_id_train : start_ids_train) {
            model.forward(ids_all, start_id_train);
            const double sequence_loss = model.backward(ids_all, start_id_train);
            epoch_loss += sequence_loss;
        }

        epoch_loss *= loss_norm_fac_epoch;
        loss_file << epoch << "\t" << epoch_loss << std::endl;

        if constexpr (VERBOSE) {
            std::cout << "INFO: Training epoch " << epoch << " completed" << std::endl;
        }
    }

    const auto training_stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> training_time_s = training_stop - training_start;

    std::cout << "INFO: All done" << std::endl
              << "INFO: Training took " << training_time_s.count() << " s" << std::endl
              << "INFO: Loss function written to file '" LOSS_FILE << "'" << std::endl;


    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // Predict the next token
    //const auto last_token_idx = nx*nv - nv;  // (nx-1)*nv
    //std::vector<double> last_logits(nv);
    //for (auto v = decltype(nv){0}; v < nv; ++v) {
    //    last_logits[v] = logits[last_token_idx + v];
    //}
    //const auto new_id    = distance(last_logits.begin(), max_element(last_logits.begin(), last_logits.end()));
    //const auto new_token = tokenizer.decode(std::vector<size_t>{new_id});
    //std::cout << input_text << " " << new_token << std::endl;
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX



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
