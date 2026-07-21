# Baselines

Launch scripts for unlearning baselines. Each run applies one method (`--method`) to a fine-tuned model checkpoint on a pre-defined forget/retain split, then auto-evaluates.

## Pre-requisites

- A fine-tuned LoRA checkpoint (produced by `main.py --finetune`), passed via `--cache_dir`.
- A split file (`data/split_*.json`)
- The FAIRGET dataset available locally (set `HF_DATASETS_OFFLINE=1`) or via HF.

## General command

```
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

Available baseline methods:

| `--method`       | Key hyperparameters              |
|------------------|----------------------------------|
| `GA`             | -                                |
| `GAD`            | `--alpha`                        |
| `RL`             | `--alpha`                          |
| `SimNPO`         | `--alpha` `--beta`                         |
| `PV7` *(OURS)*            | `--coef`, `--gamma`, `--target_layers` |
| `MIU_INTERLEAVED`| `--mine_lr`,`--mine_steps` |                   |
| `LUNAR`          | `--coef`, `--lunar_lr`, `--lunar_num_epochs`, `--target_layers`, `--module` |


### GA

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.0002 \
 --target_size 256 \
 --method GA \
 --ray_tune_report
```

---

### GAD

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.00002 \
 --target_size 256 \
 --method GAD \
 --ray_tune_report \
 --alpha 0.5
```

---

### RL

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.00001 \
 --target_size 256 \
 --method RL \
 --ray_tune_report \
 --alpha 0.5
```

---

### FAUN

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.00025 \
 --target_size 256 \
 --method PV7 \
 --ray_tune_report \
 --coef 7 \
 --gamma 50 \ 
 --target_layers 27
```

---

### MIU

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.00005 \
 --target_size 256 \
 --method MIU_INTERLEAVED \
 --ray_tune_report \
 --mine_lr  0.01 \
 --mine_steps 2000
```

---

### SimNPO

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.0001 \
 --target_size 256 \
 --method SimNPO \
 --ray_tune_report \
 --alpha 0.5 \
 --beta 1.7
```

---

### LUNAR

```sh
accelerate launch \
 --mixed_precision bf16 main.py \
 --forget \
 --model_id Qwen/Qwen2.5-VL-7B-Instruct \
 --cache_dir <FINE_TUNED_CHECKPOINT_DIR> \
 --save_dir <OUTPUT_MODEL_DIR> \
 --eval_output_dir <OUTPUT_EVAL_DIR> \
 --hf_dataset argmaxxer/FAIRGET \
 --splits_path data/<SPLIT>.json \
 --batch_size 8 \
 --lr 0.00001 \
 --target_size 256 \
 --method LUNAR \
 --ray_tune_report \
 --coef 2.0 \
 --lunar_lr 0.01 \
 --target_layers 25 \
 --lunar_num_epochs 10 \
 --module down_proj
```

---

