# LLM basics
These notes follow the YouTube lecture notes available [here](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu).

***Transformers*** lie at the core or *some* ***large language models (LLM)***. However, not all transformers are LLMs (e.g., they can serve as image recognition and classification tools) and not all LLMs are transformers (LLMs can be based on, e.g., convolutional or recurrent neural networks). See the [foundational paper on transformers](https://arxiv.org/abs/1706.03762) for in-depth info on transformers.


## Basics
1. ***Tokenization*** The input text (sentence/paragraph) is *tokenized*, i.e., it is divided up into chunks, where each chunk may or may not correspond to a full/single word; in fact, it usually doesn't. Each chunk is then assigned a unique ID, or *token*.

2. ***Encoding / Vector embedding*** Chunks must somehow be connected semantically to each other. To that end, each of them is assigned a *vector* in a $N$-dimensional space, where $N$ is a *very* large integer ($N\!\sim\!10^5\text{--}10^6$), and the shorter the *distance* between two such vectors, the more semantically related the two tokens are. This step is performed by the **encoder**.

3. ***Decoding*** The **decoder** transforms the input text in output text *one token at a time*, i.e., recursively: it starts with the first token, then starts over again using both the encoded information and the first output token to generate the second output token, and so on.
   **NOTE:** the decoding process basically involves *training a neural network*, i.e., gradually adjusting its weigths via a gradient descent method to converge to the right solution/output.



## (Self-)attention mechanism
This is a technique used to model the semantic closeness of two words and the importance of a word relative to the other words in the sentence/paragraph without relying on the distance between these words in the token vector space. This mechanism allows capturing *long-range dependencies/correlations* between words, therefore "understanding" how much attention should be given to each word.



## Evolutions of the transformer architecture
### Bidirectional Encoder Representations of Transformers (BERTs)
To predict hidden words in a given sentence, the following steps are taken:
1. The model receives randomly hidden ("masked") words during training
2. Same tokenization-encoding-(i.e., vector embedding)-decoding steps as above
3. Finally guesses the missing word from the input

The model is good at predicting a word missing from *any location* of a text. The model is *bidirectional*, meaning it looks at the text both from left to right and viceversa: that's why BERTs are good at understanding the context a word is placed in and are used for *sentiment analysis*.


### Generative Pre-trained Transformers (GPTs)
To predict hidden words in a given sentence, the following steps are taken:
1. Tokenization
2. Decoding to predict the missing word without previous encoding
   **NOTE:** the price to pay for the missing encoder blocks is a large number of decoding blocks and a huge number of input parameters (i.e., weigths of the neural network) to be optimized.

Here, you predict *one word at a time from left to right*, so the model is good at predicting the *last* word of a sentence, but not intermediate words. The model is *unidirectional*: it only looks at the text from left to right.

During training, the sentence is split in an input-only part and in a progressively predicted part. Using the first part of the sentence, you predict the next word and progressively minimize the difference between that and the actual next word, then move on to the next word once converged. In this sense, GPTs are an example of **unsupervised/self-supervised learning**: the data used for training are part of the dataset to be analyzed. They are also **auto-regressive** models, i.e., the generated output is used as part of the the input for the next prediction.

GPTs exhibit **emergent behaviors**: although only trained for next-word predictions, it also develops other skills (e.g., translating between languages, generate multiple-choice questions, ...) they are not trained for. Why this happens is still an open question.
