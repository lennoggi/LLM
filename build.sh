#!/bin/bash

set -e
set -x

rm -rf build
rm -rf install
mkdir build
cd build
cmake ..
make -j install
cd ..
rm -rf build
