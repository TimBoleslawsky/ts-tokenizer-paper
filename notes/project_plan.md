# Project Plan

- Stage 1, Setup and Preparation: (Week 1-5, Deadline: 20.02.)
  - Create a project plan. (Deadline: 13.02.)
  - Create a working environment. 
  - Train a downstream ML model, representative of a defined subset of industry-relevant tasks,
  on uncompressed data to quantify loss in predictive utility.
    - => What downstream tasks? Goal: Find and define an anomaly task (which data/ columns/ ... to use) and build a small RNN model as baseline.
    - Anomaly Detection or Fuel Consumption is probably good. We need to wait for feedback from Szabolcs here.
  - Implement a baseline model based on established neural compression frameworks such as https://www.sciencedirect.com/science/article/pii/S1568494623008153:
    -  => What framework to use? Maybe TCN RNN Autoencoder:    
    The TCN–RNN autoencoder is a representative of state-of-the-art continuous-latent neural time-series compression (discrete-latent compression methods barely exist for time-series data. And if they do (Deep Dict) they are not really reproducible). Compared to vanilla autoencoders, TCN–RNN architectures are explicitly designed for long temporal dependencies and have demonstrated stronger rate–distortion performance in recent peer-reviewed work. This allows us to evaluate our tokenizer-based approach against a competitive neural compressor rather than a toy model. This methods is also suitable because of its costly architectural design: deep temporal convolutional stacks with large receptive fields, recurrent (often sequential) decoding, attention mechanisms, and dense continuous latent representations. These components typically lead to increased FLOPs, higher activation memory, and longer inference latency compared to discrete, token-based compression methods. 
- Stage 2, Development: (Week 6-10, Deadline: 27.03.)
  - Develop a learnable tokenization module that discretizes data into semantically meaningful
  units optimized for downstream tasks.
  - Develop lightweight entropy modeling and coding schemes tailored to the tokenized representations.
  - Develop a decoder.
    - => Create a more detailed plan for the implementation ??? 
  - Write the halftime report. 
  - Write the theory part of the final report.
- Stage 3, Experiments: (Week 11-13, Deadline: 17.04.)
  - Train the model from Task 1 on the compressed data using both compression methods. 
  - Evaluate baseline model and proposed tokenization + lightweight entropy model framework based on rate-utility and computational efficiency.
    - => Define metrics in more detail.
    - Reconstruction error ???
    - Compression rate ???
    - Metrcs for "rate-utility":
      - ???
    - Metrics for "model weight":
      -	Parameter count: total number of trainable parameters
      -	Computational cost: FLOPs per fixed-length input window
      -	Inference latency: wall-clock encoding and decoding time
      -	Memory footprint: peak memory usage during inference
- Stage 4, Finalizing: (Week 14-17, Deadline: 15.05.)
  - Ablation study of the proposed tokenization + lightweight entropy model framework. 
  - Write the methodology, results, discussion, and conclusion part fo the final report. 
