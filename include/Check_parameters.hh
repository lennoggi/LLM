#ifndef CHECK_PARAMETERS_HH
#define CHECK_PARAMETERS_HH

#include "../Parameters.hh"

static_assert(NMERGES_BPE >= 0);
static_assert(VERBOSE or not VERBOSE);

#endif
