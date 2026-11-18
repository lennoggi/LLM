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
//#define INPUT_FILE  "datasets/input/SentenceCompletion.txt"

/* ------------------------------------------------
 * File storing the loss function vs training epoch
 * ------------------------------------------------ */
#define LOSS_FILE "loss.asc"


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
constexpr inline size_t BPE_MAX_VOCAB_SIZE = 1000;

/* -------------------------------------------------------------------------
 * 'end-of-word' character to be added to each word during the BPE training
 * NOTE: '\x1f' (ASCII separator) is not printable with std::cout (printing it
 *       either produces nothing or a weird character)
 * ------------------------------------------------------------------------- */
//#define BPE_END_OF_WORD "\x1f"
#define BPE_END_OF_WORD "@"


/* -------------------------
 * Number of training epochs
 * ------------------------- */
constexpr inline size_t NEPOCHS = 1000;

/* -----------------------------------------------------------------------------
 * Context window, i.e., the number of token IDs the model can read, process,
 * and remember at any given time. Training only sees these many tokens from the
 * training text at once.
 * -----------------------------------------------------------------------------*/
constexpr inline size_t CONTEXT_WINDOW = 300;

/* ---------------------------------------------------------------------------
 * The context window shift by CONTEXT_STRIDE tokens through the training text
 * - CONTEXT_STRIDE = 1                : max quality, but also max cost
 * - CONTEXT_STRIDE = CONTEXT_WINDOW   : min quality, but also min cost
 * - CONTEXT_STRIDE = CONTEXT_WINDOW/2 : balanced choice
 * --------------------------------------------------------------------------- */
constexpr inline size_t CONTEXT_STRIDE = CONTEXT_WINDOW/2;


/* -------------------------
 * Token embedding dimension
 * ------------------------- */
constexpr inline size_t DIM = 64;

/* ------------------------------------------------------------------------
 * Expansion factor for the two-layer feed-forward neural network with GELU
 * activation function used after the attention layer
 * ------------------------------------------------------------------------ */
constexpr inline size_t FFN_EXPANSION_FACTOR = 4;

/* --------------------------------------------------------------------------
 * If true, the model uses the vector embedding matrix as the logits weights.
 * TODO: comment on pros and cons
 * -------------------------------------------------------------------------- */
constexpr inline bool WEIGHT_TYING = true;


/* -----------------------------------------------------------------------------
 * Seed for the pseudo-random number generator. If non-negative, the seed will
 * be used and the output of the LLM will be reproducible; if negative, the
 * machine entropy will be used instead and results won't be reproducible.
 * NOTE: the seed should be a uint32_t . If it's too long, it will be implicitly
 *   converted into a uint32_t; if it's a float/double, it will be truncated.
 * -----------------------------------------------------------------------------*/
constexpr inline long long RANDOM_SEED = 123;
//constexpr inline long long RANDOM_SEED = -1;

/* -------------------------------------------------------------
 * Learning rate regulating the strength of the gradient descent
 * ------------------------------------------------------------- */
constexpr inline double LEARNING_RATE = 2.0;

/* ----------------------------------------------------------------
 * Regularize the variance of a token embedding vector during layer
 * normalization
 * ---------------------------------------------------------------- */
constexpr inline double LN_EPSILON = 1.e-05;

/* ----------------------------------------------------------------------------
 * Probability of dropout, i.e., of randomly setting to zero some of the
 * components of the context vectors to avoid having the model overly rely on a
 * few of these components
 * NOTE: set to a negative value to disable dropout entirely
 * ---------------------------------------------------------------------------- */
//constexpr inline double DROPOUT_PROB = -1.0;
//constexpr inline double DROPOUT_PROB = 0.1;
constexpr inline double DROPOUT_PROB = 0.0;


/* ---------
 * Verbosity
 * --------- */
constexpr inline bool VERBOSE = false;

#endif
