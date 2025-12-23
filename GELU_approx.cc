#include <cmath>
#include <vector>
#include <stdexcept>

#include "include/Declare_functions.hh"

using namespace std;


void GELU_approx(vector<double> &vec,
                 vector<double> &vec_prime) {
    const auto dim = vec.size();

    if (vec_prime.size() != dim) {
        throw runtime_error("GELU_approx(): the sizes of the two vectors must match");
        return;  // Not reached
    }

    constexpr double sqrt_2_over_pi = sqrt(2./M_PI);
    constexpr double a              = 0.044715;

    for (auto idx = decltype(dim){0}; idx < dim; ++idx) {
        const auto x    = vec.at(idx);
        const auto x2   = x*x;
        const auto th   = tanh(sqrt_2_over_pi*x*(1. + a*x2));
        const auto thp1 = th + 1.;

        vec.at(idx)       = 0.5*x*thp1;
        vec_prime.at(idx) = 0.5*(thp1 + sqrt_2_over_pi*x*(1. + 3.*a*x2)*(1. - th*th));
    }

    return;
}
