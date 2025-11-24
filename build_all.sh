#!/bin/bash

set -e
set -x

./clean_all.sh
mkdir build
cd build
cmake ..
make install
