cmake_minimum_required(VERSION 3.10)

project(LLM VERSION 1.0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(LLM Main.cc Tokenizer.cc)
target_include_directories(LLM PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_custom_target(symlink_infile ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/Input_files ${CMAKE_BINARY_DIR}/Input_files
    COMMENT "Symlinking input files into the build directory"
)
