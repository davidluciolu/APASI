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
The models are trained with SI-23k and SI-130k, named `APASI-Base-7B` and `APASI-Scaled-Scaled`, respectively.

To use the model, please follow the [code](https://github.com/haotian-liu/LLaVA/blob/main/scripts/merge_lora_weights.py) in LLaVA's official repo.

## Install
1. clone this repo
```
git clone https://github.com/davidluciolu/APASI.git
cd APASI
```
2. Create a conda environment and install the dependencies
```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip 
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install nltk 
```

## Data Preparation
1. Download the images from the COCO and VG dataset, put them into `./playground/data`
2. Download the [SI-Dataset](https://huggingface.co/datasets/lucio36/APASI-SI-dataset) and make a directory under `./playground/data/neg_data`
```
cd ./playground/data/neg_data/
mkdir detail_23k_llava-v1.5-7b_gen_llava-v1.5-7b_lvis_guide_replace_0.2_1_skip1_num1
cd detail_23k_llava-v1.5-7b_gen_llava-v1.5-7b_lvis_guide_replace_0.2_1_skip1_num1
mv [downloaded SI-23k parquet] detail_23k_llava-v1.5-7b_gen_llava-v1.5-7b_iter_0.parquet
```
3. If you want to prepare the SI data from LLaVA `detail-23k`, run the scripts
```
bash scripts/rl/make_data.sh
```

## Train
1. Run the scripts:
```
# just run dpo training
bash scripts/rl/dpo.sh

# make data + dpo
bash scripts/rl/all_in_one.sh
```

## Inference and Evaluation
1. Prepare the evaluation benchmark data following the instructions in [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 
For Object-Hal (CHAIR) evaluation, we use the sampled 500 images following [OPERA](https://github.com/shikiw/OPERA).
2. Run the script:
```
bash scripts/rl/eval_all.sh
```

## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): The baseline model for this work.
- [RLHF-V](https://github.com/RLHF-V/RLHF-V): We followed the DPO code in this repo for training.
- [OPERA](https://github.com/shikiw/OPERA): We follow the Object-Hal evaluation in this repo.

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