#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <vector>
#include <random>


void layer_norm(      std::vector<double> &vecs,
                const std::vector<double> &scale,
                const std::vector<double> &shift);

void softmax(std::vector<double> &vec);

void skip_conn_dropout(std::vector<double> &vec,
                       std::vector<double> &dropout_vec,
                       std::uniform_real_distribution<double> &udist,
                       std::mt19937 &gen);

void GELU_approx(std::vector<double> &vec);


#endif
