# ts-tokenizer

## Plan
- Stage 1, Setup and Preparation: (Week 1-5, Deadline: 20.02.)
  - Create a project plan. (Deadline: 13.02.)
  - Create a working environment. 
  - Train a downstream ML model, representative of a defined subset of industry-relevant tasks,
  on uncompressed data to quantify loss in predictive utility.
  - Implement a baseline model based on established neural compression frameworks such as https://www.sciencedirect.com/science/article/pii/S1568494623008153.
- Stage 2, Development: (Week 6-10, Deadline: 27.03.)
  - Develop a learnable tokenization module that discretizes data into semantically meaningful
  units optimized for downstream tasks.
  - Develop lightweight entropy modeling and coding schemes tailored to the tokenized representations.
  - Develop a decoder.
  - Write the halftime report. 
  - Write the theory part of the final report.
- Stage 3, Experiments: (Week 11-13, Deadline: 17.04.)
  - Train the model from Task 1 on the compressed data using both compression methods. 
  - Evaluate baseline model and proposed tokenization + lightweight entropy model framework based on rate-utility and computational efficiency.
- Stage 4, Finalizing: (Week 14-17, Deadline: 15.05.)
  - Ablation study of the proposed tokenization + lightweight entropy model framework. 
  - Write the methodology, results, discussion, and conclusion part fo the final report. 

## ToDo's
- Stage 1:
  - Setting up the project plan repo (Emrik).
  - Take everything from proposal that is appropriate (Emrik).
  - Reformulate problem and method.
      - Rough implementation plan for baseline compression technique (Tim).
      - Rough implementation plan for tokenization framework. => answer the question of how task-aware it should be.
      - Outline of the experiment setup. 
  - Add Limitations, Risk analyis, and time plan.
  - Define the downstream task: Find and define an anomaly task and/or regression (which data/ columns/ ... to use) and build a small RNN model as baseline (Tim).
  - Figure out how much of Volvos data can we show (Tim).
  - Look for an open-source dataset that can serve the same purpose (Emrik).
- Stage 2:
  - ...










