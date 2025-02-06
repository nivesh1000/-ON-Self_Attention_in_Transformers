The self-attention mechanism works in the following steps:

1. Convert each word into a vector representation.
2. Compute three different vectors for each word: **Query (Q), Key (K), and Value (V)**.
3. Calculate attention scores by multiplying Q with K.
4. Scale the scores using the square root of the dimension (dk) to stabilize gradients.
5. Apply the softmax function to get attention weights.
6. Multiply the attention weights with V to get the final self-attention output.

## Links
- [[Query Matrix]]
- [[Key Matrix]]
- [[Value Matrix]]
- [[Importance of dk in Scaling]]

## Links
- [[Self Attention]]