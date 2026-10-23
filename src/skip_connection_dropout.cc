#include <vector>
#include <random>
#include <stdexcept>

#include "declare_functions.hh"
#include "parameters.hh"


/* ========================================================================
 * Routine applying dropout to the second input vector and adding it to the
 * first one
 * ======================================================================== */
void skip_conn_dropout(std::vector<double> &vec,
                       std::vector<double> &dropout_vec,
                       std::uniform_real_distribution<double> &udist,
                       std::mt19937 &gen) {
    const auto dim = vec.size();

    if (dropout_vec.size() != dim) {
        throw std::runtime_error("skip_conn_dropout(): the sizes of the two vectors must match");
        return;  // Not reached
    }

    constexpr auto dropout_scale = 1.0/(1.0 - DROPOUT_PROB);

    for (auto idx = decltype(dim){0}; idx < dim; ++idx) {
        if constexpr (DROPOUT_PROB > 0.0) {
            const auto x = udist(gen);
            if (x < DROPOUT_PROB) {
                dropout_vec[idx] = 0.0;
            } else {
                dropout_vec[idx] *= dropout_scale;
            }
        }

        vec[idx] += dropout_vec[idx];
    }

    return;
}
