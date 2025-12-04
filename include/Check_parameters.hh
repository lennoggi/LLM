#ifndef CHECK_PARAMETERS_HH
#define CHECK_PARAMETERS_HH

#include "../Parameters.hh"

static_assert(TOKENIZER == WORD or TOKENIZER == BPE);
static_assert(BPE_MAX_VOCAB_SIZE > 0);

static_assert(DIM > 1);  // At least 2 for the variance of each token embedding vector to be well defined

static_assert(NTRAIN > 0);
static_assert(VAR_TINY > 0. and VAR_TINY < 1.);  // Should be positive, but "small"

//static_assert(CONTEXT_SIZE > 0);
static_assert(VERBOSE or not VERBOSE);

#endif
