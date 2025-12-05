#ifndef PARAMETERS_HH
#define PARAMETERS_HH


/* ----------------------------------------------
 * File containing the text used to train the LLM
 * ---------------------------------------------- */
#define INFILE_TRAINING  "Input_files/TheVerdict.txt"


/* --------------------
 * Text to be processed
 * -------------------- */
//  ***** All words in the training text *****
//#define INPUT_TEXT "The fancy bed blocked the patient"
// ***** "dog" not in the training text *****
#define INPUT_TEXT "The fancy dog blocked the patient"
// ***** Completely new text (words may or may not be in the training text *****
//#define INPUT_TEXT "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
// ***** Very long text *****
//#define INPUT_TEXT "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021. Born into a wealthy family in New York City, Trump graduated from the University of Pennsylvania in 1968 with a bachelor's degree in economics. He became the president of his family's real estate business in 1971, renamed it the Trump Organization, and began acquiring and building skyscrapers, hotels, casinos, and golf courses. He also launched side ventures, many licensing the Trump name, and filed for six business bankruptcies in the 1990s and 2000s. From 2004 to 2015, he hosted the reality television show The Apprentice, bolstering his fame as a billionaire. Presenting himself as a political outsider, Trump won the 2016 presidential election against Democratic Party nominee Hillary Clinton. During his first presidency, Trump imposed a travel ban on seven Muslim-majority countries, expanded the Mexico–United States border wall, and enforced a family separation policy on the border. He rolled back environmental and business regulations, signed the Tax Cuts and Jobs Act, and appointed three Supreme Court justices. In foreign policy, Trump withdrew the U.S. from agreements on climate, trade, and Iran's nuclear program, and initiated a trade war with China. In response to the COVID-19 pandemic from 2020, he downplayed its severity, contradicted health officials, and signed the CARES Act. After losing the 2020 presidential election to Joe Biden, Trump attempted to overturn the result, culminating in the January 6 Capitol attack in 2021. He was impeached in 2019 for abuse of power and obstruction of Congress, and in 2021 for incitement of insurrection; the Senate acquitted him both times. In 2023, Trump was found liable in civil cases for sexual abuse and defamation and for business fraud. He was found guilty of falsifying business records in 2024, making him the first U.S. president convicted of a felony. After winning the 2024 presidential election against then-Vice President Kamala Harris, he was sentenced to a penalty-free discharge, and two felony indictments against him for retention of classified documents and obstruction of the 2020 election were dismissed without prejudice. Trump began his second presidency by initiating mass layoffs of federal workers. He imposed tariffs on nearly all countries at the highest level since the Great Depression and signed the One Big Beautiful Bill Act. His administration's actions—including targeting of political opponents and civil society, actions against transgender people, deportations of immigrants, and extensive use of executive orders—have drawn over 300 lawsuits challenging the legality and constitutionality of the actions. Since 2015, Trump's leadership style and political agenda—often referred to as Trumpism—have reshaped the Republican Party's identity. Many of his comments and actions have been characterized as racist or misogynistic, and he has made false or misleading statements and promoted conspiracy theories to a degree unprecedented in American politics. Trump's actions, especially in his second term, have been described as authoritarian and contributing to democratic backsliding. After his first term, scholars and historians ranked him as one of the worst presidents in American history."


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
//#define NTRAIN 10000
#define NTRAIN 1


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
//constexpr inline double DROPOUT_PROB = -1.;
constexpr inline double DROPOUT_PROB = 0.1;


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
