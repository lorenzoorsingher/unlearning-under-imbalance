<h2 align="center"> <a href="https://arxiv.org/">Unlearning Under Imbalance: Benchmarking Fairness in Multimodal LLM Unlearning</a></h2>

<div align="center">    
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

## Quick Access

[**ArXiv**](https://huggingface.co/datasets/argmaxxer/FAIRGET) [TBD]

[**Huggingface**](https://huggingface.co/datasets/argmaxxer/FAIRGET)

[**GitHub**](https://github.com/lorenzoorsingher/unlearning-under-imbalance)


---

## Dataset

`hf download --repo-type dataset argmaxxer/FAIRGET` 

---

## Finetuning

```python
print("hello")
```
---

## Unlearning

```python
print("hello")
```

## Evaluation

```python
print("hello")
```