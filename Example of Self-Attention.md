Let's consider a simple example:

Sentence: **"The cat sat on the mat."**

1. Each word is converted into a vector.
2. Q, K, and V matrices are created for each word.
3. The dot product of Q and K gives attention scores.
4. Scores are scaled and passed through softmax.
5. Weighted sum with V gives the final output.

This allows the model to focus on important words when making predictions.

## Links
- [[Self Attention]]