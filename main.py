import argparse
import gc
import json
import os
import random
from pprint import pprint

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

import wandb
from dotenv import load_dotenv

from auto_eval import run_auto_eval
from finetune import finetune
from forget import forget
from utils import (
    MockScheduler,
    WarmUpScheduler,
)

from data_process.data_classes import train_collate_fn_qwen_mixed


def find_all_linear_names(model, args):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = [
        "multi_modal_projector",
        "vision_tower",
        "vision_model",
        "vision",
    ]

    for name, module in model.named_modules():
        if isinstance(module, cls):
            if args.lora_all_modules:
                # If all modules are to be included, add the module name directly
                lora_module_names.add(name)
                # print(f"ADDED ALL {name}")
            elif not any(mm_keyword in name for mm_keyword in multimodal_keywords):
                # print(f"ADDED {name}")
                lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    # breakpoint()
    return list(lora_module_names)


def load_model_and_processor(args, accelerator):
    """
    Load the model and processor based on the provided model_id.
    """

    if args.model_id in ["Qwen/Qwen2.5-VL-7B-Instruct"]:

        accelerator.print(f"Loading Qwen2.5 model from {args.cache_dir}")
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )

        processor = AutoProcessor.from_pretrained(
            args.cache_dir, do_image_splitting=False
        )
    elif args.model_id in ["HuggingFaceM4/Idefics3-8B-Llama3"]:

        accelerator.print(f"Loading Idefics3 model from {args.cache_dir}")

        model = AutoModelForVision2Seq.from_pretrained(
            args.cache_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(
            args.cache_dir, do_image_splitting=False
        )
    else:
        raise ValueError(f"Model {args.model_id} is not supported.")

    # Additional processor configuration if necessary
    special_tokens = ["<image>", "<pad>"]
    processor.tokenizer.padding_side = "right"  # Ensure right padding
    processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    processor.tokenizer.additional_special_tokens = list(
        set(processor.tokenizer.additional_special_tokens or []) | set(special_tokens)
    )
    processor.tokenizer.pad_token = "<pad>"

    accelerator.print(
        f"[TOKENIZER]  {processor.tokenizer.pad_token} -> {processor.tokenizer.pad_token_id}"
    )
    return model, processor


def main(args):
    print("RUNNING")

    # Load .env and configure wandb
    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
    if "WANDB_KEY" in os.environ and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]

    accelerator = Accelerator(
        log_with="wandb",
    )

    name = args.wandb_run_name or os.path.basename(args.save_dir)

    if args.finetune:
        accelerator.init_trackers(
            os.environ.get("WANDB_PROJECT", "MLLMU"),
            config=args,
            init_kwargs={
                "wandb": {
                    "name": name,
                    "mode": "offline",
                    "tags": args.tag,
                }
            },
        )

    if args.forget:
        accelerator.init_trackers(
            "FAIRGET",
            config=args,
            init_kwargs={
                "wandb": {
                    "name": name,
                    "mode": "offline",
                    "tags": args.tag,
                }
            },
        )

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        # Force synchronization
        torch.cuda.synchronize()

    if accelerator.is_main_process:
        pprint(vars(args))

    # Load or generate split
    if args.splits_path is None:
        raise ValueError(
            "[ERROR] Generate a split first and provide the path to --splits_path"
        )
    else:
        accelerator.print("[DATA] Loading precomputed split...")
        with open(args.splits_path, "r") as f:
            splits = json.load(f)

    if args.eval:
        print("[INFO] Running EVAL mode")
        if not os.path.isdir(args.save_dir):
            raise ValueError(f"[ERROR] --save_dir is not a valid directory: {args.save_dir}")
        if accelerator.is_main_process:
            checkpoint_subdirs = [
                d for d in os.listdir(args.save_dir)
                if os.path.isdir(os.path.join(args.save_dir, d))
            ]
            if len(checkpoint_subdirs) != 1:
                raise ValueError(
                    f"[ERROR] Expected exactly 1 checkpoint subfolder inside save_dir, "
                    f"found {len(checkpoint_subdirs)}: {checkpoint_subdirs}"
                )
            checkpoint_path = os.path.join(args.save_dir, checkpoint_subdirs[0])
            print(f"[INFO] Running EVAL mode on {checkpoint_path}")
            out_dir, _ = run_auto_eval(args, run_name=name, save_dir=checkpoint_path, splits=splits, model=None)
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    elif args.forget:
        print("[INFO] Running FORGET mode")
    elif args.finetune:
        print("[INFO] Running FINETUNE mode")
        args.cache_dir = (
            args.model_id
        )  # when finetuning the cache dir is huggingface model id
    else:
        raise ValueError("[ERROR] Please specify either --finetune or --forget.")

    target_size = args.target_size
    if target_size is not None:
        target_size = (target_size, target_size)

    # Build a small validation set from retain + forget test-split identities
    val_dataset = None
    args.val_retain_ids = set()
    args.val_forget_ids = set()
    if args.finetune or args.forget:

        from data_process.data_classes import IDE_eval_Dataset

        val_retain_count = min(50, len(splits["retain"]))
        val_forget_count = min(50, len(splits["forget"]))
        val_retain_ids = random.sample(splits["retain"], val_retain_count) if val_retain_count > 0 else []
        val_forget_ids = random.sample(splits["forget"], val_forget_count) if val_forget_count > 0 else []
        val_ids = val_retain_ids + val_forget_ids

        val_dataset = IDE_eval_Dataset(
            hf_dataset=args.hf_dataset,
            train_ids=val_ids,
            task="generation",
            target_size=target_size,
            media_type="text_image",
            split="test",
            log=False,
        )
        args.val_retain_ids = [int(x) for x in val_retain_ids]
        args.val_forget_ids = [int(x) for x in val_forget_ids]

    model, processor = load_model_and_processor(args, accelerator)

    accelerator.print(f"[INFO] tokenizer size: {len(processor.tokenizer)}")
    accelerator.print(
        f"[INFO] model vocab size: {model.get_input_embeddings().weight.shape[0]}"
    )
    accelerator.print(f"[INFO] micro batch size: {args.batch_size}")
    accelerator.print(
        f"[INFO] gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    accelerator.print(
        f"[INFO] total batch size: {args.batch_size * accelerator.gradient_accumulation_steps * accelerator.num_processes}"
    )
    accelerator.print(f"[INFO] learning rate: {args.lr}")
    accelerator.print(f"[INFO] {args.num_epochs} epochs")
    # NOTE: AdamW is invariant to batch size scaling, so learning rate should NOT
    # be scaled by the number of processes. If larger-batch scaling is desired,
    # use an explicit --scale_lr flag with sqrt scaling instead.

    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(processor.tokenizer))
        accelerator.print(
            f"[WARNING]: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        accelerator.print(
            f"[INFO] model vocab resized: {model.get_input_embeddings().weight.shape[0]}"
        )

    # #use reentrant gradient checkpointing for onevision compatibility
    # if args.model_id in ["HuggingFaceM4/idefics2-8b"]:
    #     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    # else:
    # was needed for onevision compatibility
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if args.lora_rank > 0:
        accelerator.print("[INFO] Applying LoRA...")
        # Setup LoRA configuration
        if args.lora_alpha == -1:
            args.lora_alpha = 2 * args.lora_rank
        find_lora_names = find_all_linear_names

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=find_lora_names(model, args),
            init_lora_weights="gaussian",
        )

        # Quantize model
        accelerator.print("Configuring PEFT model...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        accelerator.print("[INFO] Not applying LoRA.")

    # LUNAR never uses the main optimizer (it trains a local EstimatedNet
    # in Phase B and hard-copies weights in Phase C).  When LoRA is off,
    # creating a full-model AdamW would allocate ~112 GB of optimizer
    # states and OOM.  Use a dummy optimizer instead.
    if args.method == "LUNAR" and args.lora_rank == 0:
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = AdamW([dummy_param], lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.warmup == 0:
        lr_scheduler = MockScheduler(
            optimizer
        )  # Use a mock scheduler for compatibility
    else:
        lr_scheduler = WarmUpScheduler(optimizer, warmup_steps=args.warmup)

    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(
        f"[INFO] Trainable params: {trainable_params} ({trainable_params/1e6:.2f}M) | "
        f"Total params: {total_params} ({total_params/1e6:.2f}M) | "
        f"Trainable fraction: {trainable_params/total_params:.2%}"
    )

    if args.finetune:
        accelerator.print("[INFO] Finetuning...")
        model, save_dir = finetune(
            args,
            accelerator,
            model,
            optimizer,
            lr_scheduler,
            splits,
            target_size,
            train_collate_fn_qwen_mixed,
            processor
        )
        if accelerator.is_main_process and save_dir:
            out_dir, metrics = run_auto_eval(args, run_name=name, save_dir=save_dir, splits=splits, model=None)
            if metrics is not None:
                _log_eval_metrics(accelerator, metrics)
            if args.ray_tune_report and out_dir:
                _write_tune_result(out_dir, save_dir)
    elif args.forget:
        accelerator.print("[INFO] Forgetting...")
        model, save_dir = forget(
            args,
            accelerator,
            model,
            optimizer,
            lr_scheduler,
            splits,
            target_size,
            train_collate_fn_qwen_mixed,
            processor
        )
        if accelerator.is_main_process and save_dir:
            out_dir, metrics = run_auto_eval(args, run_name=name, save_dir=save_dir, splits=splits, model=None)
            if metrics is not None:
                _log_eval_metrics(accelerator, metrics)
            if args.ray_tune_report and out_dir:
                _write_tune_result(out_dir, save_dir)

    accelerator.end_training()


def _log_eval_metrics(accelerator, metrics):
    """Flatten the nested metrics dict and log to wandb via the accelerator."""
    wandb_metrics = {}
    for media in ("image", "text"):
        if media not in metrics:
            continue
        m = metrics[media]
        for group in ("retain", "forget", "protected"):
            if group in m:
                for metric_name in ("gen", "prob"):
                    val = m[group].get(metric_name, 0.0)
                    wandb_metrics[f"eval/{media}_{group}_{metric_name}"] = val
        g = m.get("global", {})
        if "mink_auc" in g:
            wandb_metrics["eval/mink_auc"] = g["mink_auc"]
        if "fairness" in g:
            wandb_metrics["eval/fairness"] = g["fairness"]
        if "fairness_v2" in g:
            wandb_metrics["eval/fairness_v2"] = g["fairness_v2"]
    if wandb_metrics:
        accelerator.log(wandb_metrics)


def _write_tune_result(eval_out_dir, save_dir):
    """Write ``tune_result.json`` for Ray Tune consumption.

    Reads ``metrics.csv`` from *eval_out_dir* and dumps key metrics alongside
    the original csv rows to a JSON file in the parent directory of *save_dir*.
    """
    import csv

    metrics_csv = os.path.join(eval_out_dir, "metrics.csv")
    if not os.path.isfile(metrics_csv):
        print(f"[TUNE-REPORT] metrics.csv not found at {metrics_csv}")
        return

    row = {}
    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            break  # first data row only

    # Build a structured result with the key metrics the sweeper cares about.
    report = {}
    for key, val in row.items():
        try:
            report[key] = float(val)
        except (ValueError, TypeError):
            report[key] = val

    report["eval_out_dir"] = eval_out_dir
    report["checkpoint_dir"] = save_dir

    result_path = os.path.join(os.path.dirname(save_dir), "tune_result.json")
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[TUNE-REPORT] tune_result.json written to {result_path}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune models with fairness adjustments"
    )

    parser.add_argument(
        "--finetune",
        action=argparse.BooleanOptionalAction,
        help="Script to run for finetuning",
    )

    parser.add_argument(
        "--forget",
        action=argparse.BooleanOptionalAction,
        help="Script to run for forgetting",
    )

    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        help="Evaluation-only mode: load model from cache_dir and run evaluation",
    )

    # COMMON ARGS ==========================================================================================
    parser.add_argument(
        "--model_id", type=str, required=True, help="Original pretrained model ID"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_model",
        help="Directory to save the model",
        required=True,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="trained model cache directory",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="argmaxxer/FAIRGET",
        help="Path to the dataset directory or HF dataset ID (e.g. 'argmaxxer/FAIRGET')",
        required=False,
    )
    parser.add_argument(
        "--splits_path", type=str, default=None, help="Path to a precomputed split file"
    )
    parser.add_argument(
        "--target_size", type=int, default=None, help="Image target size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for training",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup steps for learning rate scheduler",
    )

    parser.add_argument(
        "--media_type",
        type=str,
        default=None,
        help="Type of media (text_image or text_only)",
    )

    parser.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        help="Save checkpoints during training",
        default=False,
    )

    parser.add_argument(
        "--debug",
        type=int,
        default=-1,
        help="Debug level",
    )

    parser.add_argument(
        "--seed", type=int, default=67, help="Random seed for initialization"
    )

    # LoRa args-----------------------------------
    parser.add_argument(
        "--lora_all_modules",
        action=argparse.BooleanOptionalAction,
        help="Whether to apply LoRA to all modules or only linear layers",
        default=True,
    )

    parser.add_argument("--lora_rank", type=int, default=8, help="Size of LoRA rank")

    parser.add_argument(
        "--lora_alpha", type=int, default=-1, help="Value of LoRA alpha"
    )

    # FINETUNE ARGS ==========================================================================================

    parser.add_argument(
        "--mixed",
        type=float,
        default=0,
        help="if > 0 the dataset is mixed with a fraction of VQAv2",
    )

    # FORGET ARGS ==========================================================================================
    parser.add_argument(
        "--method",
        type=str,
        default="GA",
        choices=[
            "GA",
            "GAD",
            "PO",
            "RL",
            "SimNPO",
            "PV7",
            "MIU",
            "MIU_INTERLEAVED",
            "LUNAR",
        ],
        help="Method to use for forgetting",
    )

    ##TODO args for PV
    parser.add_argument(
        "--vectors_path",
        type=str,
        default=".",
        help="Path to the persona vectors",
    )

    parser.add_argument(
        "--coef",
        type=float,
        default=2,
        help="Coefficient for the steering vector",
    )

    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="../eval",
        help="Directory to save auto-evaluation outputs (defaults to checkpoint dir)",
    )

    parser.add_argument(
        "--target_layers",
        type=str,
        default="all",
        help="Comma-separated list of target layers for the steering vector",
    )


    parser.add_argument(
        "--use_global_avg",
        action=argparse.BooleanOptionalAction,
        help="For retain activations use the average of all identities instead of the identity specific vector (default false)",
        default=True,
    )

    ##TODO ARGS FOR GAD/PO


    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value for SCRUB method",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1,
        help="Gamma value for SCRUB method",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Beta value for NPO method",
    )

    parser.add_argument(
        "--module",
        type=str,
        default="down_proj",
        help="Which module to extract or steer lunar vectors",
    )

    parser.add_argument(
        "--lunar_lr",
        type=float,
        default=1e-2,
        help="Learning rate for LUNAR EstimatedNet training",
    )

    parser.add_argument(
        "--lunar_num_epochs",
        type=int,
        default=10,
        help="Number of epochs for LUNAR pre-training",
    )
    parser.add_argument(
        "--lunar_reg",
        type=float,
        default=0.1,
        help="Regularization coefficient for EstimatedNet weight toward original "
        "effective weight (LUNAR method). Higher values preserve retain behavior.",
    )

    # MIU args
    parser.add_argument(
        "--mine_lr",
        type=float,
        default=0.01,
        help="SGD learning rate for MINE estimator (MIU method)",
    )
    parser.add_argument(
        "--forgetting_epochs",
        type=int,
        default=-1,
        help="Epochs for MIU unlearning phase; -1 means all epochs",
    )
    parser.add_argument(
        "--mine_steps",
        type=int,
        default=100,
        help="Number of MINE update steps per epoch for MIU method",
    )

    parser.add_argument(
        "--seconds_per_forget_sample",
        type=float,
        default=0.1,
        help="Wall-clock seconds per forget sample for time-based budget; only used when BUDGET_METHOD='time'",
    )

    parser.add_argument(
        "--val_steps",
        type=int,
        default=-1,
        help="Run validation every N training steps (0 = disable)",
    )

    parser.add_argument(
        "--tag",
        type=str,
        nargs="+",
        default=[],
        help="Tags for wandb logging (can specify multiple)",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Override wandb run name (defaults to basename of --save_dir)",
    )

    parser.add_argument(
        "--ray_tune_report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write tune_result.json after auto_eval for Ray Tune consumption",
    )

    # AUTO EVAL args

    parser.add_argument(
        "--skip_gen",
        action="store_true",
        default=False,
        help="If set, skip generation evaluation",
    )

    parser.add_argument(
        "--skip_fair",
        action="store_true",
        default=False,
        help="If set, skip fairness evaluation",
    )

    parser.add_argument(
        "--skip_fairv3",
        action="store_true",
        default=False,
        help="If set, skip fairness evaluation",
    )

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    main(args)
