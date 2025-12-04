#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <vector>


void layer_norm(      std::vector<double> &vecs,
                const std::vector<double> &scale,
                const std::vector<double> &shift);

#endif
