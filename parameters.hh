#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* ----------------
 * Training dataset
 * ---------------- */
#define TRAINING_FILE  "datasets/training/TheVerdict.txt"

/* -----------------------------
 * Input dataset to be processed
 * ----------------------------- */
//#define INPUT_FILE  "datasets/input/TokenizerTest1.txt"
//#define INPUT_FILE  "datasets/input/TokenizerTest2.txt"
//#define INPUT_FILE  "datasets/input/LoremIpsum.txt"
//#define INPUT_FILE  "datasets/input/Trump.txt"
#define INPUT_FILE  "datasets/input/SentenceCompletion.txt"


/* ----------------------
 * Tokenizer
 * Choices: "WORD", "BPE"
 * --------------------- */
// ***** DON'T TOUCH *****
#define WORD 0
#define BPE  1
// ***********************
#define TOKENIZER BPE


/* ------------------------------------------------------------------
 * Maximum vocabulary size for the byte-pair encoding (BPE) tokenizer
 * ------------------------------------------------------------------ */
#define BPE_MAX_VOCAB_SIZE 200

/* -------------------------------------------------------------------------
 * 'end-of-word' character to be added to each word during the BPE training
 * NOTE: '\x1f' (ASCII separator) is not printable with std::cout (printing it
 *       either produces nothing or a weird character)
 * ------------------------------------------------------------------------- */
//#define BPE_END_OF_WORD "\x1f"
#define BPE_END_OF_WORD "@"


/* -----------------------------------------------------------------------------
 * Seed for the pseudo-random number generator. If positive, the seed will be
 * used and the output of the LLM will be reproducible; if negative, the machine
 * entropy will be used instead and results won't be reproducible.
 * NOTE: the seed should be a uint32_t . If it's too long, it will be implicitly
 *   converted into a uint32_t; if it's a float/double, it will be truncated.
 * -----------------------------------------------------------------------------*/
#define RANDOM_SEED 123
//#define RANDOM_SEED -1


/* -------------------------
 * Token embedding dimension
 * ------------------------- */
#define DIM 5


/* ----------------------------------------
 * Number of training iterations ("epochs")
 * ---------------------------------------- */
#define NTRAIN 10000
//#define NTRAIN 1


/* --------------------------------------------------------------------------
 * Set the variance of the elements of a token embedding vector to this small
 * value if that variance is exactly zero
 * -------------------------------------------------------------------------- */
#define VAR_TINY 1.e-05


/* ----------------------------------------------------------------------------
 * Probability of dropout, i.e., of randomly setting to zero some of the
 * components of the context vectors to avoid having the model overly rely on a
 * few of these components
 * NOTE: set to a negative value to disable dropout entirely
 * ---------------------------------------------------------------------------- */
//constexpr inline double DROPOUT_PROB = -1.0;
//constexpr inline double DROPOUT_PROB = 0.1;
constexpr inline double DROPOUT_PROB = 0.0;


/* ------------------------------------------------------------------------
 * Expansion factor for the two-layer feed-forward neural network with GELU
 * activation function used after the attention layer
 * ------------------------------------------------------------------------ */
#define FFN_EXPANSION_FACTOR 4


/* -------------------------------------------------------------
 * Learning rate regulating the strength of the gradient descent
 * ------------------------------------------------------------- */
#define LEARNING_RATE 0.02


/* --------------------------------------------------------------
 * Small tolerance value used to stabilize the calculation of the
 * pre-final-layer-normalization, normalized input values
 * -------------------------------------------------------------- */
#define TOLERANCE 1.e-12


/* -----------------------------------------------------------------------------
 * Context size, i.e., the number of token IDs used to predict the next token ID
 * during training
 * -----------------------------------------------------------------------------*/
//#define CONTEXT_SIZE 5


/* ---------
 * Verbosity
 * --------- */
#define VERBOSE false


#endif
