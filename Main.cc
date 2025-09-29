#include <iostream>
#include <fstream>
#include <sstream>

#include "Declarations.hh"
#include "Parameters.hh"

using namespace std;


int main() {
    ifstream infile(INFILE, ifstream::in);

    if (infile.is_open()) {
        ostringstream oss;
        oss << infile.rdbuf();
        const auto &tokens = encode(oss.str());

        for (const auto &token : tokens) {
            cout << token << endl;
        }
    }

    else {
        cerr << "Error opening file '" << INFILE << "'" << endl;
        return 1;
    }

    return 0;
}
