#ifndef PARAMETERS_HH
#define PARAMETERS_HH


#define INFILE_TRAINING  "Input_files/TheVerdict.txt"

//#define INPUT_TEXT       "The fancy bed blocked the patient"
#define INPUT_TEXT       "The fancy dog blocked the patient"  // "dog" not in the training text

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
