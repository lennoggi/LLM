#!/bin/bash

set -euox pipefail

rm -rf build
rm -rf install
mkdir build
cd build
cmake ..
make -j install
cd ..
rm -rf build
