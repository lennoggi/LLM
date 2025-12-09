#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <vector>
#include <random>


void GELU_approx(std::vector<double> &vec);

std::vector<size_t> get_max_indices(const std::vector<double> &vec,
                                    const size_t              &N);

void layer_norm(      std::vector<double> &vecs,
                const std::vector<double> &scale,
                const std::vector<double> &shift);

void skip_conn_dropout(std::vector<double> &vec,
                       std::vector<double> &dropout_vec,
                       std::uniform_real_distribution<double> &udist,
                       std::mt19937 &gen);

void softmax(std::vector<double> &vec);


#endif
