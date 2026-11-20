#ifndef CHECK_PARAMETERS_HH
#define CHECK_PARAMETERS_HH

#include "../parameters.hh"

static_assert(TOKENIZER == WORD or TOKENIZER == BPE);
static_assert(BPE_MAX_VOCAB_SIZE > 0);

static_assert(NEPOCHS > 0);
static_assert(VALIDATION_FRACTION > 0.0 and VALIDATION_FRACTION < 1.0);
static_assert(CONTEXT_WINDOW > 0);
static_assert(CONTEXT_STRIDE > 0 and CONTEXT_STRIDE <= CONTEXT_WINDOW);

static_assert(DIM > 0);  // At least 2 for the variance of each token embedding vector to be well defined
static_assert(FFN_EXPANSION_FACTOR > 0);
static_assert(WEIGHT_TYING or not WEIGHT_TYING);

static_assert(LEARNING_RATE > 0.0);
static_assert(LN_EPSILON > 0.0 and LN_EPSILON < 1.0);   // Should be positive, but "small"
static_assert(DROPOUT_PROB <= 1.0);                     // Negative means dropout is disabled

//static_assert(CONTEXT_SIZE > 0);
static_assert(VERBOSE or not VERBOSE);

#endif  // CHECK_PARAMETERS_HH
