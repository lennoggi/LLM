#ifndef PARAMETERS_HH
#define PARAMETERS_HH


#define INFILE_TRAINING  "Input_files/TheVerdict.txt"

// All words in the training text
//#define INPUT_TEXT "The fancy bed blocked the patient"
// "dog" not in the training text
//#define INPUT_TEXT "The fancy dog blocked the patient"
// Completely new text (words may or may not be in the training text
#define INPUT_TEXT "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

// Maximum vocabulary size for the byte-pair encoding (BPE) tokenizer
#define BPE_MAX_VOCAB_SIZE 200

/* 'end-of-word' character (ASCII separator) to be added to each word during
 * the BPE training
 * NOTE: usually not printable with std::cout (print either produces
 *       nothing or weird strings)                                              */
//#define BPE_END_OF_WORD "\x1f"
#define BPE_END_OF_WORD "@"

#define VERBOSE false


#endif
