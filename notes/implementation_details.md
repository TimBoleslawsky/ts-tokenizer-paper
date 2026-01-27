## Canonical Lossy Compression Pipeline

x
 → Encoder (transform)
 → z (continuous latent)
 → Quantizer
 → q (discrete symbols)
 → Entropy model P(q)
 → Entropy coder
 → bitstream

## Implementation Details

| Compression Stage | TCRAE (TCN–RNN AE) | Tokenization-based Compression |
|------------------|-------------------|--------------------------------|
| Input signal | Continuous time series | Continuous time series |
| Transform coding | Deep TCN encoder + RNN | Lightweight learned encoder |
| Continuous latent `z` | High-precision, dense | Intermediate embeddings |
| Explicit quantization | ❌ Absent (implicit only) | ✔️ Present (tokenizer / codebook) |
| Discrete symbols `q` | ❌ None | ✔️ Finite token vocabulary |
| Entropy modeling | ❌ Not used | ✔️ Learned symbol distribution |
| Entropy coding | ❌ Not used | ✔️ Arithmetic / ANS coding |
| Compression control | Architectural bottleneck | Token granularity + entropy |
| Output representation | Latent tensors | Bitstream |
| Compression metric | Proxy (latent size) | True bits per sample |
| Pipeline completeness | Partial (transform only) | Full neural codec |

### Detailed Implementation: TCRAE (TCN–RNN Autoencoder)

1. Input handling and preprocessing
	•	Multivariate time-series data are segmented into fixed-length windows.
	•	Each channel is normalized independently using statistics computed on the training set.
	•	Windows are treated as independent samples during training and evaluation.

⸻

2. TCN encoder (transform coding stage)
	•	The encoder consists of a stack of Temporal Convolutional Network (TCN) blocks with increasing dilation factors.
	•	Each TCN block contains:
	•	A 1D convolution operating along the temporal dimension
	•	Normalization (BatchNorm or LayerNorm)
	•	A nonlinear activation function (ReLU or LeakyReLU)
	•	Optional dropout for regularization
	•	A residual connection to stabilize training
	•	Temporal downsampling is introduced via strided convolutions or pooling layers to reduce sequence length.
	•	The encoder maps the input signal to a lower-resolution, higher-dimensional latent sequence, forming the continuous bottleneck.

Outcome:
A compressed continuous latent representation whose size is controlled by the downsampling factor and channel width.

⸻

3. RNN bottleneck (sequence compression)
	•	The encoder output is processed by a recurrent neural network (LSTM or GRU).
	•	The RNN captures longer-range temporal dependencies that are not easily modeled by convolutions alone.
	•	The hidden state dimensionality and number of layers define the strength of the bottleneck.
	•	The RNN produces a sequence of hidden states (or a reduced representation) that serves as the core compressed signal.

Interpretation:
Compression is achieved implicitly by restricting temporal resolution and latent dimensionality rather than by discretization.

⸻

4. RNN decoder
	•	A symmetric RNN decoder reconstructs a latent sequence from the bottleneck representation.
	•	The decoder mirrors the bottleneck RNN in depth and hidden size.
	•	No probabilistic modeling or entropy estimation is performed at this stage.

⸻

5. TCN decoder (signal reconstruction)
	•	A mirrored TCN stack upsamples the latent sequence back to the original temporal resolution.
	•	Transposed convolutions or interpolation-based upsampling are used.
	•	A final linear projection maps features back to the original channel dimension.

Training objective:
	•	Reconstruction loss (e.g., MSE or MAE) between input and reconstructed signal.

⸻

6. Compression characteristics (TCRAE)
	•	Compression rate is defined structurally via:
	•	Temporal downsampling factor
	•	Latent dimensionality
	•	No explicit quantization, entropy modeling, or bitstream generation is performed.
	•	Reported compression rates are proxy metrics derived from latent size.

⸻

### Detailed Implementation: Tokenization-Based Compression

1. Input handling and preprocessing
	•	Identical preprocessing and windowing as used for TCRAE to ensure comparability.
	•	Normalization statistics are shared across methods.

⸻

2. Lightweight encoder (feature extraction)
	•	A shallow neural encoder (e.g., small CNN or MLP applied per timestep) maps the input signal to intermediate embeddings.
	•	The encoder is intentionally kept lightweight to avoid introducing heavy representational power before tokenization.
	•	Temporal resolution is preserved or mildly reduced, depending on the tokenizer design.

⸻

3. Vector quantization / tokenization
	•	Intermediate embeddings are discretized using a learned codebook.
	•	Each embedding vector is replaced by the index of its nearest codebook entry.
	•	This step introduces the only explicit source of information loss.
	•	The output is a sequence of discrete token IDs drawn from a finite vocabulary.

Outcome:
A symbolic representation of the time series suitable for storage, transmission, or direct consumption by downstream models.

⸻

4. Entropy modeling
	•	A learned probabilistic model estimates the distribution of token sequences.
	•	Contextual dependencies between tokens may be modeled autoregressively or using lightweight temporal context.
	•	The entropy model defines the expected bitrate of the representation.

⸻

5. Entropy coding
	•	Tokens are compressed into a bitstream using a standard entropy coder (e.g., arithmetic coding or ANS).
	•	Frequently occurring tokens receive shorter codes, resulting in efficient compression.
	•	The resulting bitstream represents the final compressed output.

⸻

6. Compression characteristics (tokenization)
	•	Compression rate is measured in true bits per sample or bits per second.
	•	Rate is controlled via:
	•	Codebook size
	•	Token sequence length
	•	Entropy model capacity
	•	The method implements a complete neural compression pipeline.


## Comparison of Implementations
Problem: What does “compression rate” mean for AE-based methods like TCRAE?

Autoencoder-based time-series compression methods such as TCRAE do not produce discrete symbols or bitstreams. Instead, they “compress” by reducing the dimensionality and temporal resolution of the data through a continuous bottleneck. As a result, the compression rates reported in these works are proxy metrics based on latent size (e.g., downsampling factor and latent dimensionality), not true entropy-coded bitrates. This creates an ambiguity when comparing them to tokenization-based methods that report actual bits per sample.

Proposed solution: Use aligned proxy rates and make assumptions explicit

We address this mismatch by:
	1.	Using the compression ratios reported by TCRAE as structural proxy rates, following the conventions of the AE-based literature.
	2.	Matching our tokenizer-based method to comparable effective compression levels (e.g., same latent dimensionality or equivalent rate budget).
	3.	Explicitly stating that TCRAE’s compression rate reflects dimensionality reduction rather than entropy-coded bits, and interpreting it as an upper bound on the bitrate of continuous-latent methods without entropy coding.

This enables a fair, controlled comparison while remaining faithful to the assumptions and evaluation protocols of both compression paradigms.

### Note on downstream evaluation and implementation comparability

- Autoencoder-based compression methods such as TCRAE are conventionally evaluated by decoding latent representations back into the signal domain and measuring reconstruction fidelity or downstream task performance on reconstructed signals.
- In contrast, tokenization-based compression methods are typically designed to produce discrete representations that are consumed directly by downstream models without explicit reconstruction.
- To account for this difference, we distinguish between:
  - Reconstruction-based evaluation, in which both methods are decoded and evaluated using an identical downstream model, serving as a controlled baseline; and
  - Representation-based evaluation, in which compressed representations (continuous latents or discrete tokens) are fed directly into a shared downstream model.
- In the representation-based setting, minimal and symmetric adapter layers are employed to map different representation types into a common input space, while keeping the core downstream architecture identical.
- This dual evaluation strategy balances experimental control with practical relevance, enabling fair comparison while reflecting realistic deployment scenarios for compressed time-series representations.

### Lightweight Adapter Design 

To compare compressed representations fairly, we use minimal adapters whose sole purpose is to align input formats, not to learn new features.

Core idea:
Both continuous AE latents and discrete tokens are reduced to a single fixed-size vector per sample, which is then fed into the same downstream model.

⸻

Adapter for AE (continuous latents)
	•	The autoencoder outputs a short sequence of real-valued vectors (e.g., 100 time steps × 32 values).
	•	We average over time, resulting in a single vector (32 values).
	•	This vector is passed through one linear layer to match the downstream model’s input size (e.g., 32 → 64).

Interpretation:
We summarize “what the AE remembers” about the signal, without adding any new structure or intelligence.

⸻

Adapter for token-based compression
	•	The tokenizer outputs a sequence of token IDs (e.g., 100 tokens).
	•	Each token is mapped to a small vector via a standard embedding lookup (e.g., 100 → 32 values).
	•	We average these embeddings over time, just like for the AE.
	•	The result is passed through the same type of linear layer (32 → 64).

Interpretation:
We summarize “which symbols occurred and how often,” again without learning temporal patterns.

⸻

Why this comparison is fair
	•	Both adapters:
	•	Use one pooling operation
	•	Use one linear layer
	•	Add no temporal modeling
	•	Have comparable parameter counts
	•	The downstream model (for regression or classification) is identical in all cases.

The adapters act like type converters, not feature extractors.

⸻

Downstream tasks (kept intentionally simple)
	•	Regression (fuel consumption):
A single linear layer predicting one value.
	•	Classification (anomaly detection):
A single linear layer predicting normal vs. anomalous.

This ensures that any performance difference comes from the compressed representation, not from downstream model capacity.

