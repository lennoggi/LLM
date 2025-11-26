#ifndef CHECK_PARAMETERS_HH
#define CHECK_PARAMETERS_HH

#include "../Parameters.hh"

static_assert(BPE_MAX_VOCAB_SIZE > 0);
static_assert(VERBOSE or not VERBOSE);

#endif
