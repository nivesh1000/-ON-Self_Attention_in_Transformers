
The self-attention mechanism is a key component of transformer models, allowing them to focus on different parts of the input sequence dynamically. It helps capture long-range dependencies between words in a sentence by computing attention scores that determine how much influence each word should have on the others.

## Steps of Self-Attention

1. **Convert each word into a vector representation**
   - Each word in the input sequence is represented as an embedding vector, which captures its semantic meaning.
   - The input sentence is transformed into a matrix \( X \), where each row corresponds to the embedding of a word.

2. **Compute three different vectors for each word: Query (Q), Key (K), and Value (V)**
   - These vectors are obtained by multiplying the input embeddings with learned weight matrices:
     \[ Q = XW_Q, K = XW_K, V = XW_V \]
   where \(X\) is the input matrix, and \(W_Q, W_K, W_V\) are the trainable weight matrices that learn to capture different aspects of the words' relationships.

3. **Calculate attention scores by multiplying Q with K**
   - The dot product of each query with all keys produces a similarity score, indicating how relevant each word is to another:
     \[ \text{Scores} = QK^T \]
   - Higher scores indicate stronger relationships between words.

4. **Scale the scores using the square root of the dimension (d_k) to stabilize gradients**
   - Since the dot product values can be large, dividing by \( \sqrt{d_k} \) ensures more stable gradients and prevents excessively large softmax outputs:
     \[ \text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}} \]
   - This prevents some words from dominating the attention distribution.

5. **Apply the softmax function to get attention weights**
   - The softmax function normalizes the scores into probabilities, ensuring they sum to 1 across each row:
     \[ \text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \]
   - This step ensures that the model attends more to important words while ignoring less relevant ones.

6. **Multiply the attention weights with V to get the final self-attention output**
   - The attention weights are used to weigh the value vectors, producing a new representation of each word:
     \[ \text{Output} = \text{Attention Weights} \times V \]
   - This output is then passed to the next layer in the transformer model.


## Links
- [[Query Matrix]]
- [[Key Matrix]]
- [[Value Matrix]]
- [[Importance of dk in Scaling]]

## Links
- [[Self Attention]]