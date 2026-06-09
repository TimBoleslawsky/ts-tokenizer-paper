## Paper Outline: UniEdgeCodec as the Main Contribution

Goal: present UniEdgeCodec (partitioned RVQ + signal-specific losses) as a compact, reproducible architecture that improves reconstruction of heterogeneous signals and yields more stable training. Formalize and empirically quantify the observed "stability vs freedom" trade-off as a secondary analysis supported by ablations.

**Title:** UniEdgeCodec: Partitioned Vector-Quantized Compression for Heterogeneous Vehicle Time Series Data

**1. Introduction**
- Problem statement: edge heterogeneous telemetry; trade-off between CR, reconstruction, and downstream utility.
- Limitations of single shared codebooks and naive reconstruction losses.
- Our contributions (explicit bullets): UniEdgeCodec design, empirical gains, stability analysis + practical guidelines.

**2. Related Work**
- Time Series Neural compression 
- Vector quantization & residual VQ + Audio/Speech Neural Compression
- Edge-Optimized Time Series Neural Compression + EdgeCodec
- Multimodality: Prior uni-codebook / partitioning ideas.
    - UniCodec => Manual partitioning of the codebook with annotated training data + domain-conditioned latent representations through MoE before quantization.
    - https://arxiv.org/pdf/2201.12904 => Specialized for specifically handling representational.heterogeneity.  
- Heterogeneous signal compression as a consequence of multimodality (Validation of claim: A New Characterization of Rain and Clouds: Results from a Statistical Inversion of Count Data). 
    - https://arxiv.org/pdf/2305.16416 => More in terms of distributed neural compression. 

**3. Method: UniEdgeCodec**
- Architecture diagram (encoder, partitioned RVQ, decoder, per-channel routing)
- Main architectural decisions: 
    - How do we partition the codebook & assign the signals to these partitions? 
        - Manually - like we already do and like they do in UniCodec. 
        - More sophisticated: 
            1. We want to cluster by statistical similarity and assess how "difficult" each signal cluster is, which corresponds to how big the partition of the codebook should be. 
            2. We want some underlying overall codebook so that each signal type can share common characteristics. => This is more true to what "residuals" are supposed to do compared to what UniCodec does. In the UniCodec architecture every domain has to learn from scratch. They combat this a bit with the MoE, but we can't do that because of computational efficiency concerns. 
        - Idea:
            - Run warmup using the shared codebook for N warmup epochs. After warmup epochs gather the latents. 
            - Run global K-means clustering of latents and compute "cluster difficulty" (could be reconstruction error per cluster or cluster entropy). 
            - Allocate capacity based on "cluster difficulty". 
            - Initialize codebooks using two-stage shared + partitioned RVQ: First the shared codebook that captures common structure (as used in the warmup) and second N partitioned codebook layers that quantize signal-specific residual details.
            - Route signals based on metadata collected from first run. 
- Implementation details for reproducibility (codebook sizes, embedding dims, seeds, training schedule).

**4. Experimental Setup**
- Datasets: `Volvo EMOB`, `Volvo Test Fleet`, `Public Automotive Telemetry Dataset (maybe comma2k19)`
- Baseline: `EdgeCodec` 
- Metrics: CR, MSE, corr., downstream utility retention, stability (per-seed std, convergence epochs).

**5. Results**
- Main table: CR | MSE | Corr | Utility | Std across seeds (UniEdgeCodec vs EdgeCodec)
- Per-signal examples (smooth vs discontinuous recon plots)
- Stability summary (boxplots of per-seed metrics; convergence behavior)

**6. Discussion**
- Interpret trade-offs and when to prefer partitioning vs shared codebook
- Practical recommendations for edge deployments (codebook budgeting, signal grouping guidelines)
- Limitations and next steps.

**7. Conclusion**
- Short summary and takeaway design rules.

### Questions
- EdgeCodec has adversarial training. We ignored that for the thesis (too difficult to tune). What do we do here? 
- Should we focus on purely automotive data and remove the appliances energy use case and add a public vehicle use case?
- How do we justify the CR we use? Possibly test different ones? Probably from a range of 10:1 - 50:1 - 100:1 - 200:1. 

### Notes
- I think multimodality is a bit unclear. We actively work on statistical (and structural) heterogeneity which is a consequence of multimodality. 
