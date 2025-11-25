#ifndef PARAMETERS_HH
#define PARAMETERS_HH


#define INFILE_TRAINING  "Input_files/TheVerdict.txt"

//#define INPUT_TEXT       "The fancy bed blocked the patient"
#define INPUT_TEXT       "The fancy dog blocked the patient"  // "dog" not in the training text

/* How many times the most common char/multichar pair within the traning text is
 * identified, i.e., how many 'best' char/multichar merges are performed        */
#define NMERGES_BPE 1000

/* 'end-of-word' character (ASCII separator) to be added to each word during
 * the BPE training
 * NOTE: usually not printable with std::cout (print either produces
 *       nothing or weird strings)                                              */
//#define END_OF_WORD_BPE "\0x1f"
#define END_OF_WORD_BPE "@"

#define VERBOSE false


#endif
