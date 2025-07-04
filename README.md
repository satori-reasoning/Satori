# Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search
[![Homepage](https://img.shields.io/badge/🏠-Homepage-3C47EB.svg)](https://satori-reasoning.github.io/) &nbsp;&nbsp; [![HuggingFace](https://img.shields.io/badge/🤗-Model&Demo-E87948.svg)](https://huggingface.co/Satori-reasoning) &nbsp;&nbsp; [![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://arxiv.org/pdf/2502.02508) &nbsp;&nbsp;

This repository contains the official implementation of `Satori-Qwen-7B` in the paper [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://arxiv.org/pdf/2502.02508) (ICML 2025).


## News 🎉
- **[2025/06/02]** We have released our RL training code to help the community reproduce our work!
- **[2025/05/01]** Our paper has been accepted to present at ICML 2025!
- **[2025/02/04]** We have released our model and data through [HuggingFace](https://huggingface.co/Satori-reasoning).

## **Introduction**

Satori is a 7B parameter LLM that can **autoregressive search** for reasoning, allowing it to self-reflect and self-explore alternative strategies without external guidance. Built on **Qwen-2.5-Math-7B**, Satori achieves state-of-the-art reasoning performance through small-scale **Format Tuning (FT)** and large-scale **self-improvement via reinforcement learning (RL)**.

## Key Features

- **Autoregressive Search Capabilities:** Self-reflection and self-exploration without external feedback.

- **Chain-of-Action-Thought (COAT) Reasoning:**
  Leverage several meta-action tokens to guide reasoning,

  - **Continue Reasoning** (<|continue|>): encourages the LLM to build upon its current reasoning trajectory by generating the next intermediate step. 
  - **Reflect** (<|reflect|>): prompts the model to pause and verify the correctness of prior reasoning steps.
  - **Explore Alternative Solution** (<|explore|>): signals the model to identify critical flaws in its reasoning and explore a new solution.

- **Transferability**: Train on math domain, but generalize to unseen domains beyond math.

## Training Framework

### **1. Format Tuning (FT) via Imitation Learning**

- Use a **multi-agent data synthesis framework** (Generator, Critic, Reward Model) to create **COAT-style** demonstration trajectories.
- Train the base model to imitate the COAT reasoning format.

### **2. Self-Improvement via Reinforcement Learning (RL)**

- **Restart and Explore (RAE)**: Train the model to reason from intermediate states to encourage deeper reflection.
- **Iterative Self-Improvement**: Alternate between RL training and policy distillation for iterative improvements.


## Quick Start 🔧

### Installation

To train our model with OpenRLHF, first install the environment with:
```bash
cd Satori
pip install -e .
```

### Model and Data
- Download model checkpoints: [SFT Model Ckpt](https://huggingface.co/Satori-reasoning/Satori-SFT-7B), [Outcome Reward Model](https://huggingface.co/Satori-reasoning/Satori-RM-7B).
- Download RL training data: [RL Data with RAE](https://huggingface.co/datasets/Satori-reasoning/Satori_RL_data_with_RAE).
### Training Script
```bash
bash examples/satori/train.sh
```

## **Evaluation**

### **Math Reasoning Evaluation**

Satori-Qwen-7B achieves SOTA performance and outperforms Qwen-2.5-Math-7B-Instruct which uses the same base model (Qwen-2.5-Math-7B). After round 2 training, Satori-Qwen-7B (Round 2)
demonstrates even stronger performance on hard tasks.

| Scale     | Model                        | GSM8K | MATH500 | OlymBench | AMC2023 | AIME2024 | AVG.     |
| --------- | ---------------------------- | ----- | ------- | --------- | ------- | -------- | -------- |
| **Large** | Llama-3.1-70B-Instruct       | 94.1  | 68.0    | 29.4      | 42.5    | 13.3     | 49.5     |
|           | OpenMath2-Llama3.1-70B       | 94.1  | 71.8    | 30.1      | 45.0    | 13.3     | 50.9     |
|           | QwQ-32B-Preview              | 95.5  | 90.6    | 61.2      | 77.5    | 50.0     | 75.0     |
| **Small** | Llama-3.1-8b-Instruct        | 84.4  | 51.9    | 15.1      | 22.5    | 3.3      | 35.4     |
|           | OpenMath2-Llama3.1-8B        | 90.5  | 67.8    | 28.9      | 37.5    | 6.7      | 46.3     |
|           | NuminaMath-7B-CoT            | 78.9  | 54.6    | 15.9      | 20.0    | 10.0     | 35.9     |
|           | Qwen-2.5-7B-Instruct         | 91.6  | 75.5    | 35.5      | 52.5    | 6.7      | 52.4     |
|           | Qwen-2.5-Math-7B-Instruct    | 95.2  | 83.6    | 41.6      | 62.5    | 16.7     | 59.9     |
|           | **Satori-Qwen-7B**           | 93.2  | 85.6    | 46.6      | 67.5    | 20.0     | **62.6** |
|           | **Satori-Qwen-7B (Round 2)** | 93.9  | 83.6    | 48.5      | 72.5    | 23.3     | **64.4** |

### **General Domain Reasoning Benchmarks**
Trained **only on math** datasets, Satori-Qwen-7B exhibits strong transferability across diverse out-of-domain reasoning benchmarks and outperforms Qwen-2.5-Math-7B-Instruct by a large margin. 
Moreover, despite not being trained in other domains, Satori-Qwen-7B achieves performance comparable to or exceeding other small-scale general instruct models.

| Scale     | Model                        | FOLIO | BGQA | CRUXEval | StrategyQA | TableBench | STEM | Avg.     |
| --------- | ---------------------------- | ----- | ---- | -------- | ---------- | ---------- | ---- | -------- |
| **Large** | Llama-3.1-70B-Instruct       | 65.0  | 58.3 | 59.6     | 88.8       | 34.2       | 61.7 | 61.3     |
|           | OpenMath2-Llama3.1-70B       | 68.5  | 68.7 | 35.1     | 95.6       | 46.8       | 15.1 | 55.0     |
|           | QwQ-32B-Preview              | 84.2  | 71.1 | 65.2     | 88.2       | 51.5       | 71.3 | 71.9     |
| **Small** | Llama-3.1-8b-Instruct        | 63.5  | 50.3 | 38.5     | 92.2       | 32.4       | 43.4 | 53.4     |
|           | OpenMath2-Llama3.1-8B        | 57.1  | 49.0 | 11.1     | 84.4       | 34.2       | 10.9 | 41.1     |
|           | NuminaMath-7B-CoT            | 53.2  | 44.6 | 28.0     | 77.8       | 29.1       | 11.3 | 40.7     |
|           | Qwen-2.5-7B-Instruct         | 72.4  | 53.0 | 58.1     | 91.3       | 43.2       | 57.1 | **62.5** |
|           | Qwen-2.5-Math-7B-Instruct    | 68.9  | 51.3 | 28.0     | 85.3       | 36.2       | 45.2 | 52.5     |
|           | **Satori-Qwen-7B**           | 71.4  | 61.8 | 42.5     | 86.3       | 43.4       | 56.7 | 60.4     |
|           | **Satori-Qwen-7B (Round 2)** | 72.9  | 58.5 | 41.1     | 90.4       | 44.6       | 57.4 | **60.8** |



## **Satori Team Members**
### **Core Contributors**
- [Maohao Shen](https://maohaos2.github.io/Maohao/), MIT
- [Guangtao Zeng](https://chaoscodes.github.io/), SUTD
- [Zhenting Qi](https://zhentingqi.github.io/), Harvard
### **Contributors**
   \*: Project lead
- Zhang-Wei Hong, MIT
- Zhenfang Chen, MIT-IBM Watson AI Lab
- Wei Lu, SUTD
- Gregory W. Wornell, MIT
- Subhro Das, MIT-IBM Watson AI Lab
- David Cox, MIT-IBM Watson AI Lab
- Chuang Gan\*, UMass, MIT-IBM Watson AI Lab

## **Contact Information**
For questions, please:
- Raise an issue in our GitHub repository
- Contact us at:
  - [satori2025@outlook.com](mailto:satori2025@outlook.com)


## **Citation**
```
@misc{shen2025satorireinforcementlearningchainofactionthought,
      title={Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search}, 
      author={Maohao Shen and Guangtao Zeng and Zhenting Qi and Zhang-Wei Hong and Zhenfang Chen and Wei Lu and Gregory Wornell and Subhro Das and David Cox and Chuang Gan},
      year={2025},
      eprint={2502.02508},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02508}, 
}
```
