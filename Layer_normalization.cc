#include <cassert>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "include/Declare_functions.hh"
#include "Parameters.hh"

using namespace std;


/* =========================================================================
 * Routine normalizing the components of each vector in a list of vectors so
 * that these components average out to 0 and their variance is 1. Then, the
 * vector components are scaled and shifted.
 * ========================================================================= */
void layer_norm(      vector<double> &vecs,
                const vector<double> &scale,
                const vector<double> &shift) {
    /* Specifying 'long int' explicitly here so that the compiler knows it has
     * to pick std::div(long int a, long int b) (std::div_t is overloaded)      */
    const long int ntot     = vecs.size();
    const long int vec_size = scale.size();

    if (shift.size() != vec_size) {
        throw runtime_error("layer_normalization(): shift.size() must equal scale.size(), which in turn must equal the size of each of the input vectors");
    }

    const auto nvecs_rem = std::div(ntot, vec_size);
    const auto nvecs     = nvecs_rem.quot;

    if (nvecs_rem.rem != 0) {
        throw runtime_error("layer_normalization(): inconsistent vector size leads to non-integer number of input vectors");
    }


    for (auto m = decltype(nvecs){0}; m < nvecs; ++m) {
        const auto idx_m = m*vec_size;
        double mean      = 0.; 
        double sum_diffs = 0.; 

        /* Welford's algorithm to compute the mean amd variance of a
         * sample in one pass and without a potential catastrophic
         * cancellation when computing the variance                     */
        for (auto i = decltype(vec_size){0}; i < vec_size; ++i) {
            const auto mi     = idx_m + i;
            const auto delta1 = vecs.at(mi) - mean;
                       mean  += delta1/static_cast<double>(i+1);
            const auto delta2 = vecs.at(mi) - mean;
                   sum_diffs += delta1*delta2;
        }

        assert(sum_diffs >= 0.);

        /* NOTE: sum_diffs==0 can only happen if all elements in
         *       vecs.at(mi) are the same, which is very unlikely     */
        assert(sum_diffs >= 0.);
        constexpr auto sqrt_var_inv_fallback = 1./std::sqrt(static_cast<double>(VAR_TINY));
        const     auto sqrt_var_inv          = (sum_diffs == 0.) ? sqrt_var_inv_fallback : std::sqrt(static_cast<double>(vec_size-1)/sum_diffs);
        assert(sqrt_var_inv > 0.);

        for (auto i = decltype(vec_size){0}; i < vec_size; ++i) {
            const auto mi = idx_m + i;
            vecs.at(mi) = scale.at(i)*(sqrt_var_inv*(vecs.at(mi) - mean)) + shift.at(i);
        }
    }

    return;
}
