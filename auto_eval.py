import argparse
import json
import os
import random
from datetime import datetime

from eval_metrics import compute_run_metrics
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForVision2Seq
)

from data_process.data_classes import (
    _get_assistant_delimiter_tokens,
    IDE_eval_Dataset,
    IDE_fairness_Dataset,
    eval_collate_fn_qwen,
)


def _load_model(args, save_dir):
    """Load the merged model from the saved checkpoint directory."""
    if args.model_id in ["Qwen/Qwen2.5-VL-7B-Instruct"]:
        print("[AUTO-EVAL] Loading Qwen2.5-VL model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            save_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
    elif args.model_id in ["HuggingFaceM4/Idefics3-8B-Llama3"]:
            print("[AUTO-EVAL] Loading Idefics3 model...")

            model = AutoModelForVision2Seq.from_pretrained(
                    save_dir,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    attn_implementation="flash_attention_2",
                )
    else:
        raise ValueError(f"[AUTO-EVAL] Unsupported model: {args.model_id}")

    model.eval()
    return model


def _load_processor(save_dir):
    """Load the processor from the saved checkpoint (offline-safe)."""
    print(f"[AUTO-EVAL] Loading processor from {save_dir}...")
    processor = AutoProcessor.from_pretrained(
        save_dir, padding_side="left", local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    return processor


def _run_eval_loop(
    args, model, processor, dataset, out_file, device, batch_size, max_tokens=50, fairness_eval=False
):
    """Core evaluation loop: generates responses, computes scores, writes jsonl."""
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda x: eval_collate_fn_qwen(x, processor),
        num_workers=4,
    )

    with open(out_file, "w", encoding="utf-8") as f:
        pass

    pad_id = processor.tokenizer.pad_token_id

    for data in tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Eval {os.path.basename(out_file)}",
    ):
        batch, batch_mia, prompts, gts, IDs, attributes, metadatas = data

        if batch is not None:
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **batch.to(device),
                        max_new_tokens=max_tokens,
                        output_logits=True,
                        return_dict_in_generate=True,
                    )
            except RuntimeError as e:
                print(f"[AUTO-EVAL] Error {e} skipping batch.")
                torch.cuda.empty_cache()
                continue

            generated_ids = outputs.sequences
            batch_size = generated_ids.shape[0]
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            batch_size = batch_mia["input_ids"].shape[0] if batch_mia is not None else len(IDs)
            generated_text = [""] * batch_size

        if batch_mia is not None:
            try:
                with torch.no_grad():
                    outputs_mia = model(**batch_mia.to(device))
            except RuntimeError as e:
                print(f"[AUTO-EVAL] Error {e} skipping batch MIA score.")
                torch.cuda.empty_cache()
                mink_scores = [None] * batch_size
                probability = [None] * batch_size
            else:
                logits = outputs_mia.logits

                seq_resps = []
                ids_resps = []

                for idx, (seq_data, ids_mia, meta) in enumerate(zip(logits, batch_mia["input_ids"], metadatas)):
                    if batch is not None:
                        resp_len = (batch["input_ids"][idx] != pad_id).sum() - (ids_mia != pad_id).sum()
                    else:
                        delim = _get_assistant_delimiter_tokens(processor).to(device)
                        response_len = 0
                        for i in range(len(ids_mia) - len(delim) + 1)[::-1]:
                            if torch.equal(ids_mia[i:i + len(delim)], delim):
                                resp_start_abs = i + len(delim)
                                non_pad_before = resp_start_abs - (ids_mia[:resp_start_abs] == pad_id).sum()
                                total_non_pad = (ids_mia != pad_id).sum()
                                response_len = (total_non_pad - non_pad_before).item()
                                break
                        resp_len = -response_len

                    ids_resp = ids_mia[resp_len:]
                    seq_resp = seq_data[resp_len - 1 : -1]

                    seq_resps.append(seq_resp)
                    ids_resps.append(ids_resp)

                mink_scores = []
                probability = []
                for idx, (seq_data, ids_data, meta) in enumerate(zip(seq_resps, ids_resps, metadatas)):
                    ids_data = ids_data.unsqueeze(-1)
                    probs = F.softmax(seq_data, dim=-1)
                    log_probs = F.log_softmax(seq_data, dim=-1)
                    token_log_probs = log_probs.gather(
                        dim=-1, index=ids_data
                    ).squeeze(-1)
                    mu = (probs * log_probs).sum(-1)
                    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

                    mink_plus = (token_log_probs - mu) / sigma.sqrt()
                    k_length = 4
                    topk = np.sort(mink_plus.float().cpu())[:k_length]
                    mink_scores.append(np.mean(topk).item())

                    if fairness_eval:
                        last_tkns = meta["first_diff_token_index"]
                        last_id = ids_data[last_tkns].squeeze(dim=0)                            
                        probability.append(probs[last_tkns][last_id].item())
                    else:
                        prob_score = torch.cumprod(
                            probs.gather(dim=-1, index=ids_data).squeeze(-1), dim=0
                        )[-1] ** (1 / probs.shape[0])
                        probability.append(prob_score.item())

                


        else:
            mink_scores = [None] * batch_size
            probability = [None] * batch_size

        for prompt, gt, id, gen, attribute, prob, mink in zip(
            prompts, gts, IDs, generated_text, attributes, probability, mink_scores
        ):
            record = {
                "ID": id,
                "prompt": prompt,
                "gen_text": gen,
                "ground_truth": gt,
                "attribute": attribute,
                "probability": prob,
                "mink_score": mink,
            }

            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        
        if args.debug > 2:
            break

    print(f"[AUTO-EVAL] Records saved to {out_file}")


def run_auto_eval(args, run_name, save_dir, splits, model=None):
    """Main entry point for post-unlearning auto-evaluation.

    Runs on a single GPU (no Accelerator).  If *model* is given it is used
    directly (must already be merged and in eval mode); otherwise the fresh
    checkpoint is loaded from *save_dir*.
    """
    print("[AUTO-EVAL] Starting automatic evaluation...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = "".join(random.choices("abcdefghijklmnopq0123456789", k=4))
    run_name = f"{run_name}_{timestamp}_{random_suffix}"

    base_dir = getattr(args, "eval_output_dir", None) or os.path.dirname(save_dir)
    out_dir = os.path.join(base_dir, run_name, "run_0")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[AUTO-EVAL] Results will be saved to: {out_dir}")

    splits_out = os.path.join(out_dir, "splits.json")
    with open(splits_out, "w") as f:
        json.dump(splits, f, indent=4)

    args_out = os.path.join(out_dir, "args.json")
    with open(args_out, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[AUTO-EVAL] Args saved to: {args_out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = _load_model(args, save_dir)
        model = model.to(device)
        cleanup_model = True
    else:
        model.eval()
        cleanup_model = False

    processor = _load_processor(save_dir)

    ide_ids = splits["retain"] + splits["forget"]
    target_size = args.target_size
    if target_size is not None:
        target_size = (target_size, target_size)

    print(f"[AUTO-EVAL] Evaluating {len(ide_ids)} identities...")

    # --- Text-Image Generation (test split) ---

    if not args.skip_gen:
        dataset_ti = IDE_eval_Dataset(
            hf_dataset=args.hf_dataset,
            train_ids=ide_ids,
            task="generation",
            target_size=target_size,
            media_type="text_image",
            split="test",
            log=True,
        )
        out_file = os.path.join(out_dir, "generation_text_image.jsonl")
        _run_eval_loop(
            args, model, processor, dataset_ti, out_file, device, args.batch_size,
        )

    # --- Fairness Evaluation ---

    if not args.skip_fair:
        target_protected = splits.get("target_protected", [])
        if len(target_protected) == 2:
            target_attr = target_protected[0].split("+")[0]
            protected_attr = target_protected[1].split("+")[0]
        else:
            target_attr = "all"
            protected_attr = "all"
        print(
            "[AUTO-EVAL] Running fairness evaluation: "
            f"target={target_attr}, protected={protected_attr}"
        )

        dataset_fair = IDE_fairness_Dataset(
            hf_dataset=args.hf_dataset,
            target_size=(256, 256),
            target_attribute=target_attr,
            protected_attribute=protected_attr,
            log=True,
        )
        out_file = os.path.join(out_dir, "fairness_text_image.jsonl")
        _run_eval_loop(
            args, model, processor, dataset_fair, out_file, device, args.batch_size,
        )


    if cleanup_model:
        del model
        torch.cuda.empty_cache()

    # --- Compute metrics from the jsonl outputs ---
    metrics = None
    try:
        from eval_metrics import compute_run_metrics, write_metrics_csv

        metrics = compute_run_metrics(out_dir, hf_dataset=args.hf_dataset) #TODO fix dataset path
        csv_path = write_metrics_csv(metrics, run_name, out_dir)
        print(f"[AUTO-EVAL] Metrics CSV saved to {csv_path}")
    except Exception as e:
        print(f"[AUTO-EVAL] Metrics computation skipped: {e}")

    print("[AUTO-EVAL] Automatic evaluation complete.")
    return out_dir, metrics


def get_parser():
    parser = argparse.ArgumentParser(
        description="Standalone auto-evaluation on a saved checkpoint"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Original pretrained model ID (e.g. Qwen/Qwen2.5-VL-7B-Instruct)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to the saved checkpoint directory (merged model with processor)",
    )

    parser.add_argument(
        "--splits_path",
        type=str,
        required=True,
        help="Path to splits.json defining retain/forget/target_protected",
    )

    parser.add_argument(
        "--hf_dataset",
        type=str,
        required=True,
        help="Path to the dataset directory or HF dataset ID (e.g. 'argmaxxer/FAIRGET')",
    )

    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="./eval_output",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Image target size (e.g. 256)",
    )

    parser.add_argument(
        "--debug",
        type=int,
        default=-1,
        help="Debug level",
    )

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


def main():
    parser = get_parser()
    args = parser.parse_args()

    # if not os.path.isdir(args.save_dir):
    #     raise ValueError(f"--save_dir is not a valid directory: {args.save_dir}")

    with open(args.splits_path, "r") as f:
        splits = json.load(f)

    run_name = os.path.basename(os.path.normpath(args.save_dir))
    out_dir, metrics = run_auto_eval(args, run_name=run_name, save_dir=args.save_dir, splits=splits)
    print(f"[AUTO-EVAL] Done. Results: {out_dir}")
    if metrics:
        from pprint import pprint
        pprint(metrics)


if __name__ == "__main__":

    main()


# python auto_eval.py \
# --mode_id Qwen/Qwen2.5-VL-7B-Instruct \
# --save_dir Qwen/Qwen2.5-VL-7B-Instruct  \
# --splits_path  data/split_lowadult50.json

