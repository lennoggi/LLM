#include <vector>
#include <random>
#include <stdexcept>

#include "include/Declare_functions.hh"
#include "Parameters.hh"

using namespace std;


/* ========================================================================
 * Routine applying dropout to the second input vector and adding it to the
 * first one
 * ======================================================================== */
void skip_conn_dropout(vector<double> &vec,
                       vector<double> &dropout_vec,
                       uniform_real_distribution<double> &udist,
                       mt19937 &gen) {
    const auto dim = vec.size();

    if (dropout_vec.size() != dim) {
        throw runtime_error("skip_conn_dropout(): the sizes of the two vectors must match");
        return;  // Not reached
    }

    constexpr auto dropout_scale = 1./(1. - DROPOUT_PROB);

    for (auto idx = decltype(dim){0}; idx < dim; ++idx) {
        if constexpr (DROPOUT_PROB > 0.) {
            const auto x = udist(gen);
            if (x < DROPOUT_PROB) {
                dropout_vec.at(idx) = 0.;
            } else {
                dropout_vec.at(idx) *= dropout_scale;
            }
        }

        vec.at(idx) += dropout_vec.at(idx);
    }

    return;
}
