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

## Comparison of Implementations
Problem: What does “compression rate” mean for AE-based methods like TCRAE?

Autoencoder-based time-series compression methods such as TCRAE do not produce discrete symbols or bitstreams. Instead, they “compress” by reducing the dimensionality and temporal resolution of the data through a continuous bottleneck. As a result, the compression rates reported in these works are proxy metrics based on latent size (e.g., downsampling factor and latent dimensionality), not true entropy-coded bitrates. This creates an ambiguity when comparing them to tokenization-based methods that report actual bits per sample.

⸻

Proposed solution: Use aligned proxy rates and make assumptions explicit

We address this mismatch by:
	1.	Using the compression ratios reported by TCRAE as structural proxy rates, following the conventions of the AE-based literature.
	2.	Matching our tokenizer-based method to comparable effective compression levels (e.g., same latent dimensionality or equivalent rate budget).
	3.	Explicitly stating that TCRAE’s compression rate reflects dimensionality reduction rather than entropy-coded bits, and interpreting it as an upper bound on the bitrate of continuous-latent methods without entropy coding.

This enables a fair, controlled comparison while remaining faithful to the assumptions and evaluation protocols of both compression paradigms.
