#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "include/Declare_functions.hh"

using namespace std;


vector<size_t> get_max_indices(const vector<double> &vec,
                               const size_t         &N) {
    const auto vec_size = vec.size();

    if (vec_size == 0) {
        throw runtime_error("get_max_indices(): input vector must have at least one element");
    } else {
        vector<size_t> max_indices(vec_size);

        for (auto i = decltype(vec_size){0}; i < vec_size; ++i) {
            max_indices.at(i) = i;
        }

        const auto compare = [&vec](size_t a, size_t b) {
            const auto vec_a = vec.at(a);
            const auto vec_b = vec.at(b);
            if (vec_a == vec_b) { return a < b; }                  // Tie braker
            else                { return vec_a > vec_b; }  // Typical case
        };

        if (N >= vec_size) {
            cerr << "WARNING ( get_max_indices() ): max " << N << " indices requested from input vector of size "
                 << vec_size << ". Returning all the indices of the input vector sorted by descending value." << endl;

            sort(max_indices.begin(), max_indices.end(), compare);
        } else {
            // Have the first N indices be larger than all the other ones
            nth_element(max_indices.begin(), max_indices.begin() + N,
                        max_indices.end(), compare);

            sort(max_indices.begin(), max_indices.begin() + N, compare);
            max_indices.resize(N);
        }

        return max_indices;
    }
}
