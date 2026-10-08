cmake_minimum_required(VERSION 3.10)

project(LLM VERSION 1.0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXE "llm")

add_executable(${EXE}
    src/bpe_tokenizer.cc
    src/file_to_string.cc
    src/gelu_approx.cc
    src/layer_normalization.cc
    src/main.cc
    src/skip_connection_dropout.cc
    src/softmax.cc
    src/word_tokenizer.cc
)

target_include_directories(${EXE} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

add_custom_target(symlink_training_files ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/training_files ${CMAKE_BINARY_DIR}/training_files
    COMMENT "Symlinking training files into the build directory"
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

install(TARGETS ${EXE}
        RUNTIME DESTINATION bin
)
