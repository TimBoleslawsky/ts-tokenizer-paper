
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

### Metrics
    - Reconstruction error ???
    - Compression rate ???
    - Metrcs for "rate-utility":
      - ???
    - Metrics for "model weight":
      -	Parameter count: total number of trainable parameters
      -	Computational cost: FLOPs per fixed-length input window
      -	Inference latency: wall-clock encoding and decoding time
      -	Memory footprint: peak memory usage during inference

This ensures that any performance difference comes from the compressed representation, not from downstream model capacity.
