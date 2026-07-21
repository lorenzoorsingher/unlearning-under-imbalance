<h2 align="center"> <a href="https://github.com/lorenzoorsingher/unlearning-under-imbalance">Unlearning Under Imbalance: Benchmarking Fairness in Multimodal LLM Unlearning</a></h2>

<div align="center">

#### [Lorenzo Orsingher](https://github.com/lorenzoorsingher), [Thomas De Min](https://scholar.google.com/citations?user=fnh_i0cAAAAJ&hl=en), [Massimiliano Mancini](https://scholar.google.com/citations?hl=it&authuser=1&user=bqTPA8kAAAAJ),</br> [Davide Talon](https://scholar.google.com/citations?user=IiMwp7EAAAAJ&hl=) and [Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ&hl=it&authuser=1) 

<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/XXXX.XXXXX) -->
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat-square&logo=github)](https://lorenzoorsingher.github.io/unlearning-under-imbalance/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-FAIRGET-FFD21E?style=flat-square)](https://huggingface.co/datasets/argmaxxer/FAIRGET)

<img src="./images/dataset.png" width="100%" height="50%">

</div>

## Abstract

Machine unlearning has emerged as a tool for removing personal data from trained models to comply with recent AI regulations. To evaluate unlearning effectiveness in multimodal large language models (MLLMs), prior works fine-tune models on fictitious identities, simulating unlearning requests on subsets of these IDs, which are typically uniformly distributed. However, in realistic scenarios, people from different demographic groups may request to be unlearned at different frequencies, potentially altering the model’s internal beliefs for these groups and leading to biased behaviors. To fill this gap, we propose FAIRGET, the first Visual Question Answering benchmark that evaluates unlearning under unbalanced, realistic, forget requests. These requests are designed to simulate multiple realistic scenarios, ranging from simple to challenging settings, that lead to biased unlearned models if fairness is not accounted for. Additionally, we propose FAUN, the first unlearning algorithm for MLLMs that forgets unlearning data while preserving model fairness. FAUN exploits a bias-aware activation steering mechanism to unlearn identities while accounting for the unbalanced nature of the forget data. Experiments on FAIRGET and the established FIUBench demonstrate our method's superiority both in unlearning quality and fairness.

---

## Installation

```
conda create -n myenv python=3.11
conda activate myenv
pip install -r requirements.txt
pip install flash-attn==2.6.1 --no-build-isolation
``` 

---

## Finetuning

For fine-tuning a new retain-only model (gold-standard baseline):

```bash
accelerate launch --mixed_precision bf16 main.py \
  --finetune \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --save_dir <OUTPUT_DIR>/model \
  --eval_output_dir <OUTPUT_DIR>/eval \
  --splits_path data/<SPLIT>.json \
  --batch_size 16 \
  --lr 2e-5 \
  --lora_rank 64 \
  --lora_all_modules \
  --num_epochs 2 \
  --target_size 256 \
  --warmup 500 \
  --mixed 0.15
```

---

## Unlearning

The general command follows the pattern of:

```python
accelerate launch --mixed_precision bf16 main.py \
  --forget \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
  --save_dir <OUTPUT_MODEL_DIR> \
  --eval_output_dir <OUTPUT_EVAL_DIR> \
  --hf_dataset argmaxxer/FAIRGET \
  --splits_path data/<SPLIT>.json \
  --batch_size 8 \
  --lr <LR> \
  --target_size 256 \
  --method <METHOD> \
  --ray_tune_report
```
more examples are available in [BASELINES.md](./BASELINES.md) 

The final results will have this structure:

```
OUT_DIR/<run_name>
├── eval
│   └── <EVAL_ID>
│       └── run_0
│           ├── args.json                     # args for run and eval
│           ├── fairness_text_image.jsonl     # fairness eval outputs
│           ├── generation_text_image.jsonl   # generation eval outputs
│           ├── metrics.csv                   # metrics results
│           └── splits.json                   # data split for the run
└── model
    ├── <RUN_ID>
    │   └── checkpoint files...
    └── tune_result.json                      # run recap with results
```

## Evaluation

Evaluation can be ran standalone with:

```python
python auto_eval.py \
--model_id Qwen/Qwen2.5-VL-7B-Instruct \
--save_dir Qwen/Qwen2.5-VL-7B-Instruct or <MODEL_DIR>  \
--hf_dataset argmaxxer/FAIRGET \
--splits_path  data/split_lowadult50.json \
--eval_output_dir <OUTPUT_EVAL_DIR>
```