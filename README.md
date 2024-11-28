## Explanation of Seq2Seq Model in Soft Terms

The provided code implements a sequence-to-sequence (Seq2Seq) model using an encoder-decoder architecture. Below is the mathematical foundation of each component and its interplay during the forward pass.

---

### **1. Encoder**

The encoder maps an input sequence $$\(\mathbf{X} = (x_1, x_2, \dots, x_T)\) of length \(T\)$$ into a context vector containing hidden states that summarize the sequence.

#### **Input Transformation**
1. **Embedding Layer**:
   $$\mathbf{E}_t = \text{Embedding}(x_t), \quad \mathbf{E}_t \in \mathbb{R}^{d_{\text{emb}}}$$
   $$where \(x_t\) is the integer token at timestep \(t\), and \(d_{\text{emb}}\)$$ is the dimensionality of the embedding vector.

3. **LSTM Layer**:
   The embedded sequence \((\mathbf{E}_1, \mathbf{E}_2, \dots, \mathbf{E}_T)\) is passed into an LSTM:
   \[
   (\mathbf{h}_t, \mathbf{c}_t) = \text{LSTM}(\mathbf{E}_t, (\mathbf{h}_{t-1}, \mathbf{c}_{t-1}))
   \]
   where:
   - \(\mathbf{h}_t \in \mathbb{R}^{d_{\text{hidden}}}\) is the hidden state at timestep \(t\),
   - \(\mathbf{c}_t \in \mathbb{R}^{d_{\text{hidden}}}\) is the cell state at timestep \(t\),
   - \(d_{\text{hidden}}\) is the dimensionality of the hidden and cell states.

4. **Output**:
   At the final timestep \(T\), the encoder outputs the hidden and cell states:
   \[
   (\mathbf{h}_T, \mathbf{c}_T)
   \]

---

### **2. Decoder**

The decoder generates an output sequence \(\mathbf{Y} = (y_1, y_2, \dots, y_L)\) of length \(L\), one token at a time, conditioned on the encoder's context vector \((\mathbf{h}_T, \mathbf{c}_T)\) and the previously generated token.

#### **Step-by-Step Process**
1. **Input Transformation**:
   Each token \(y_{t-1}\) is embedded:
   \[
   \mathbf{E}'_{t-1} = \text{Embedding}(y_{t-1}), \quad \mathbf{E}'_{t-1} \in \mathbb{R}^{d_{\text{emb}}}
   \]

2. **LSTM Layer**:
   The decoder LSTM updates its hidden and cell states:
   \[
   (\mathbf{h}_t, \mathbf{c}_t) = \text{LSTM}(\mathbf{E}'_{t-1}, (\mathbf{h}_{t-1}, \mathbf{c}_{t-1}))
   \]

3. **Output Prediction**:
   The decoder generates logits for the vocabulary at each timestep:
   \[
   \mathbf{o}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o, \quad \mathbf{o}_t \in \mathbb{R}^{V}
   \]
   where:
   - \(V\) is the size of the output vocabulary,
   - \(\mathbf{W}_o \in \mathbb{R}^{V \times d_{\text{hidden}}}\) and \(\mathbf{b}_o \in \mathbb{R}^V\) are learnable weights and biases.

4. **Prediction**:
   The final token probabilities are obtained using the softmax function:
   \[
   p(y_t | y_{<t}, \mathbf{X}) = \text{softmax}(\mathbf{o}_t)
   \]

---

### **3. Seq2Seq Forward Pass**

The Seq2Seq model ties the encoder and decoder to process both the source sequence \(\mathbf{X}\) and target sequence \(\mathbf{Y}\).

#### **1. Encoding**
- The encoder processes the source sequence to generate the context vector:
  \[
  (\mathbf{h}_T, \mathbf{c}_T) = \text{Encoder}(\mathbf{X})
  \]

#### **2. Decoding with Teacher Forcing**
- The decoder generates the target sequence using teacher forcing with a ratio \(\gamma \in [0, 1]\):
  \[
  y_t = 
  \begin{cases} 
  \text{argmax}_y \, p(y_t | y_{<t}, \mathbf{X}), & \text{with probability } (1-\gamma) \\
  \text{teacher-provided } y_t, & \text{with probability } \gamma
  \end{cases}
  \]

- For each timestep \(t\), the decoder updates:
  \[
  (\mathbf{h}_t, \mathbf{c}_t) = \text{LSTM}(\mathbf{E}'_{t-1}, (\mathbf{h}_{t-1}, \mathbf{c}_{t-1}))
  \]

#### **3. Output Sequence**
- The final output tensor \(\mathbf{O}\) aggregates predictions for all timesteps:
  \[
  \mathbf{O} \in \mathbb{R}^{B \times L \times V}, \quad \mathbf{O}_{i,j,k} = p(y_k | y_{<j}, \mathbf{X}_i)
  \]
  where \(B\) is the batch size.

---

### **4. Loss Function**

To optimize the model, the Cross-Entropy Loss is applied at each timestep \(t\), comparing the predicted distribution \(p(y_t | y_{<t}, \mathbf{X})\) with the ground truth token \(y_t\):
\[
\mathcal{L} = -\frac{1}{B \times L} \sum_{i=1}^B \sum_{t=1}^L \log p(y_t^{(i)} | y_{<t}^{(i)}, \mathbf{X}^{(i)})
\]

---

### **Highlights**
- **Encoder**: Encodes temporal information via \(\mathbf{h}_T, \mathbf{c}_T\).
- **Decoder**: Autoregressively generates outputs, leveraging both the encoder's context and its own predictions.
- **Teacher Forcing**: Balances learning efficiency and robustness by mixing true targets and predictions during training.
- **Optimization**: Loss is minimized using gradient descent, updating all trainable parameters.

By this, the Seq2Seq model is capable of learning complex input-output relationships for tasks such as machine translation or sequence prediction.
