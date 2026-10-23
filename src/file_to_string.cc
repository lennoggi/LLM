#include <sstream>
#include <fstream>

std::string file_to_string(const std::string &filename) {
    std::ifstream file(filename, std::ifstream::in);

    if (file.is_open()) {
        std::ostringstream file_ss;
        file_ss << file.rdbuf();
        return file_ss.str();
    } else {
        std::ostringstream err_ss;
        err_ss << "Unable to read file '" << filename<< "'";
        throw std::runtime_error(err_ss.str());
        return std::string("");  // Not reached
    }
}
