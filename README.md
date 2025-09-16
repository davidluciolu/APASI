# APASI: Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations

[[📖 Paper](https://arxiv.org/abs/2509.11287)] [[🤗 APASI-Model](https://huggingface.co/collections/lucio36/apasi-model-68c52dfb1103aba8c675a756)] [[🤗 SI-Dataset](https://huggingface.co/datasets/lucio36/APASI-SI-dataset)] 

## Introduction
This is the official implementation of our paper: 
**Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations**. 

In this work, we propose **A**utonomous **P**reference **A**lignment via **S**elf-**I**njection (**APASI**).
Unlike previous methods relying on annotations from human or external AI models,
APASI leverages the target LVLM itself to self-inject hallucinations into a generated response, 
creating a pair of responses with varying preference levels for DPO-based preference alignment.

Our work is accepted by **EMNLP 2025**.

![](./assets/comparison.png)

## Dataset
We present the preference dataset, [SI-Dataset](https://huggingface.co/datasets/lucio36/APASI-SI-dataset),
which is constructed using only the target LVLM.
Specifically, we construct the **SI-23k**, deriving from images and descriptive responses
in the detail-23k subset of the [LLaVA's instruction tuning data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/detail_23k.json).
The scaled-up **SI-130k** is constructed by adding unannotated images from the VisualGenome (VG) dataset

## APASI Model Weight
We release the [LoRA adaptation weights](https://huggingface.co/collections/lucio36/apasi-model-68c52dfb1103aba8c675a756)
of APASI based on [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b). 
The models are trained with SI-23k and SI-130k, named `APASI-Base-7B` and `APASI-Scaled-7B`, respectively.

To use the model, please follow the [code](https://github.com/haotian-liu/LLaVA/blob/main/scripts/merge_lora_weights.py) in LLaVA's official repo.

## Code
Coming soon...

[//]: # (## Install)

[//]: # ()
[//]: # (## Data Preparation)

[//]: # ()
[//]: # (## Train)

[//]: # ()
[//]: # (## Inference)

## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): The baseline model for this work.
- [RLHF-V](https://github.com/RLHF-V/RLHF-V): We followed the DPO code in this repo for training.

## Citation
If you find our model/code/data/paper helpful, please consider cite our papers 📝 and star us ⭐️！

```bibtex
@misc{lu2025mitigatinghallucinationslargevisionlanguage,
      title={Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations}, 
      author={Yifan Lu and Ziqi Zhang and Chunfeng Yuan and Jun Gao and Congxuan Zhang and Xiaojuan Qi and Bing Li and Weiming Hu},
      year={2025},
      eprint={2509.11287},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.11287}, 
}
```