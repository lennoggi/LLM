#include <cmath>
#include <vector>
#include <stdexcept>

#include "declare_functions.hh"
#include "parameters.hh"


/* =============================================================================
 * Routine normalizing the components of each vector in a list of vectors
 * (flattened matrix!) so that these components average out to 0 and their
 * variance is 1. Then, the vector components are scaled and shifted. The
 * intermediate vectors `xhat` and `sigmas_inv` are output so they don't have to
 * be recomputed during the backward training step.
 * ============================================================================= */
void layer_norm(      std::vector<double> &x,
                const std::vector<double> &scale,
                const std::vector<double> &shift,
                      std::vector<double> &xhat,
                      std::vector<double> &sigmas_inv) {
    /* Specifying 'long int' explicitly here so that the compiler knows it has
     * to pick std::div(long int a, long int b) (std::div_t is overloaded) */
    const long int nxdim = x.size();
    const long int dim   = scale.size();

    if (xhat.size() != nxdim) {
        throw std::runtime_error("layer_norm(): xhat and x sizes differ");
        return;  // Not reached
    }

    if (shift.size() != dim) {
        throw std::runtime_error("layer_norm(): shift.size() differs from `dim` (size of each input vector)");
        return;  // Not reached
    }

    const auto nxdiv = std::div(nxdim, dim);
    const auto nx    = nxdiv.quot;

    if (nxdiv.rem != 0) {
        throw std::runtime_error("layer_norm(): inconsistent input leads to non-integer number of input vectors");
        return;  // Not reached
    }

    if (sigmas_inv.size() != nx) {
        throw std::runtime_error("layer_norm(): sigmas_inv.size() differs from nx (number of input vectors)");
        return;  // Not reached
    }


    for (auto m = decltype(nx){0}; m < nx; ++m) {
        const auto idx_m = m*dim;

        /* Welford's algorithm to compute the mean and variance of a
         * sample in one pass and without a potential catastrophic
         * cancellation when computing the variance */
        double mean      = 0.0;
        double sum_diffs = 0.0;

        for (auto i = decltype(dim){0}; i < dim; ++i) {
            const auto mi     = idx_m + i;
            const auto delta1 = x[mi] - mean;
                       mean  += delta1/static_cast<double>(i+1);
            const auto delta2 = x[mi] - mean;
                   sum_diffs += delta1*delta2;
        }

        /* NOTE: sum_diffs==0 can only happen if all elements in
         *   x[mi] are the same, which is very unlikely. However, very long
         *   sequences (i.e., a very large `dim`) can make `sum_diffs` go
         *   slightly negative due to floating point rounding in the Welford
         *   loop above. */
        sum_diffs = std::max(0.0, sum_diffs);
        const auto sigma_inv = std::sqrt(static_cast<double>(dim)/(sum_diffs + VAR_TINY));
        sigmas_inv[m] = sigma_inv;

        for (auto i = decltype(dim){0}; i < dim; ++i) {
            const auto mi = idx_m + i;
            xhat[mi] = sigma_inv*(x[mi] - mean);
            x[mi]    = scale[i]*xhat[mi] + shift[i];
        }
    }

    return;
}
