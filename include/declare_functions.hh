#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <vector>
#include <string>
#include <random>


std::string file_to_string(const std::string &filename);

void GELU_approx(std::vector<double> &vec,
                 std::vector<double> &vec_prime);

void layer_norm(      std::vector<double> &vecs,
                const std::vector<double> &scale,
                const std::vector<double> &shift,
                      std::vector<double> *sigmas_inv = nullptr);

void skip_conn_dropout(std::vector<double> &vec,
                       std::vector<double> &dropout_vec,
                       std::uniform_real_distribution<double> &udist,
                       std::mt19937 &gen);

void softmax(std::vector<double> &vec);


#endif
