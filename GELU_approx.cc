#include <cmath>
#include <vector>

#include "include/Declare_functions.hh"

using namespace std;


void GELU_approx(vector<double> &vec) {
    constexpr double sqrt_2_over_pi = sqrt(2./M_PI);
    constexpr double a              = 0.044715;

    for (auto &el : vec) {
        el = 0.5*el*(1. + tanh(sqrt_2_over_pi*el*(1. + a*el*el)));
    }

    return;
}
