#include <cassert>
#include <cmath>
#include <vector>
#include <limits>

#include "include/Declare_functions.hh"

using namespace std;


/* ===========================================================
 * Routine applying a softmax normalization to an input vector
 * =========================================================== */
void softmax(vector<double> &vec) {
    auto max = -numeric_limits<double>::infinity();

    // Find the largest element
    for (const auto &el : vec) {
        if (el > max) {
            max = el;
        }
    }

    assert(isfinite(max));
    double sum_exp = 0.; 

    for (auto &el : vec) {
        const auto exp_att = exp(el - max);
        el       = exp_att;
        sum_exp += exp_att;
    }

    assert(sum_exp > 0.);

    for (auto &el : vec) {
        el /= sum_exp;
    }

    return;
}
