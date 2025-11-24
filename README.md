# LLM
A large language model (LLM) built from scratch

## Requirements
- A C++ compiler supporting at least the `c++17` standard
- CMake; minimum required version 3.10, older version may work (edit the `cmake_minimum_required` string in `CMakeLists.txt` accordingly)

## Instructions
- Build with
  ```
  ./build_all.sh
  ```
- Run with
  ```
  ./install/bin/llm
  ```
- Remove the build and install directories with
  ```
  ./clean_all.sh
  ```

## References
Raschka, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications, 2024
