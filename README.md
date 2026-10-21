[![CI](https://github.com/lennoggi/LLM/actions/workflows/CI.yml/badge.svg)](https://github.com/lennoggi/LLM/actions/workflows/CI.yml)

# LLM
A large language model (LLM) built from scratch

## Requirements
- A C++ compiler supporting at least the `c++17` standard
- CMake; minimum required version 3.10, older versions may work (edit the `cmake_minimum_required` string in `CMakeLists.txt` accordingly)
- A library providing the `cblas.h` header file and implementing the `cblas_dgemm()` function. The code has so far only been tested with OpenBLAS.

## Instructions
- Build with
  ```
  ./build.sh
  ```
- Run with
  ```
  ./install/bin/llm
  ```

## References
Raschka, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications, 2024
