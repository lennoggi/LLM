#include <sstream>
#include <fstream>

using namespace std;


string file_to_string(const string &filename) {
    ifstream file(filename, ifstream::in);

    if (file.is_open()) {
        ostringstream file_ss;
        file_ss << file.rdbuf();
        return file_ss.str();
    } else {
        ostringstream err_ss;
        err_ss << "Unable to read file '" << filename<< "'";
        throw runtime_error(err_ss.str());
        return string("");  // Not reached
    }
}
