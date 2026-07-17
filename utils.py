import json
import os
import random
from datetime import datetime

import accelerate.utils
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


class MockScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class WarmUpScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        # Store initial LRs
        self.initial_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.initial_lrs[i] * lr_scale

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class Logger:
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def log(self, message):
        if self.accelerator.is_main_process:
            print(message)


def save_checkpoint(splits, accelerator, args, model, processor, epoch=None):

    print("[SAVING] Saving model checkpoint...")
    accelerator.wait_for_everyone()

    save_dir = None
    if accelerator.is_main_process:
        print("[SAVING] Unwrapping and preparing for saving...")
        if epoch is None:
            save_dir = prepare_folders(splits, accelerator, args)
        else:
            save_dir = prepare_folders(splits, accelerator, args, suffix=f"EP{epoch}")

        unwrapped_model = accelerator.unwrap_model(model)

        # Merge LoRA weights
        if isinstance(unwrapped_model, PeftModel):
            unwrapped_model = unwrapped_model.merge_and_unload()

        # Cast back to bf16 after merge (LoRA weights are fp32 by default)
        unwrapped_model = unwrapped_model.to(torch.bfloat16)

        # Save the model
        unwrapped_model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)  # Save processor/tokenizer
        print(f"Model saved to: {save_dir}")

        args_path = os.path.join(save_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Args saved to: {args_path}")

    print("[SAVING] Waiting for all processes to finish...")

    accelerator.wait_for_everyone()

    print("[SAVING] Checkpoint saving completed.")

    return save_dir


def prepare_folders(splits, accelerator, args, prefix="", suffix=""):

    save_dir_with_suffix = ""
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = "".join(random.choices("abcdefghijklmnopq0123456789", k=4))

        args.save_dir = f"{args.save_dir.rstrip('/')}{suffix}"
        save_dir_with_suffix = os.path.join(
            args.save_dir, f"{timestamp}_{random_suffix}"
        )
        os.makedirs(save_dir_with_suffix, exist_ok=True)

        splits_path = os.path.join(save_dir_with_suffix, "splits.json")
        with open(splits_path, "w") as f:
            json.dump(splits, f)

        visualize_splits(
            splits,
            image=True,
            savepath=os.path.join(save_dir_with_suffix, "split_distribution.png"),
        )
        print(f"[DATA] Split saved to: {splits_path}")

    return save_dir_with_suffix


def visualize_splits(splits, image=False, savepath="data/split_distribution.png"):

    retain = splits["retain"]
    forget = splits["forget"]
    group_to_ids = splits["group_to_ids"]
    id_to_group = splits["id_to_group"]

    target_protected = splits["target_protected"]

    all_ids = retain + forget

    protected_ids = set(retain + forget)
    for protected_attribute in target_protected:
        group, attribute = protected_attribute.split("+")
        protected_ids.intersection_update(set(group_to_ids[group][attribute]))

    n_retain = len(retain)
    n_forget = len(forget)
    n_total = n_retain + n_forget

    n_protected_total = len(protected_ids)
    n_protected_retain = len(set(retain) & protected_ids)
    n_protected_forget = len(set(forget) & protected_ids)

    print(f"Total size: {n_total}")
    print(f"Retain size: {n_retain} ({n_retain/n_total*100:.2f}%)")
    print(f"Forget size: {n_forget} ({n_forget/n_total*100:.2f}%)")

    if n_forget > 0:
        print(
            f"Protected ratio in forget set: {n_protected_forget}/{n_forget} ({n_protected_forget/n_forget*100:.2f}%)"
        )
        print(
            f"Protected ratio in retain set: {n_protected_retain}/{n_retain} ({n_protected_retain/n_retain*100:.2f}%)"
        )

    if image:

        subgroups = [
            n_forget - n_protected_forget,
            n_protected_forget,
            n_retain - n_protected_retain,
            n_protected_retain,
        ]
        print(subgroups)
        # Pie chart for forget-retain distribution
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 2, 1)
        plt.pie(
            subgroups,
            labels=["Forget", "Forget Protected", "Retain", "Retain Protected"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#ff9999", "#ffcc99", "#66b3ff", "#99ff99"],
        )
        plt.title("Forget-Retain Distribution")

        if n_forget != 0 and n_protected_forget != 0:
            # Pie chart for protected-non protected distribution in forget set
            plt.subplot(1, 2, 2)
            plt.pie(
                [n_protected_forget, n_forget - n_protected_forget],
                labels=["Protected", "Non-Protected"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#ffcc99", "#99ff99"],
            )
            plt.title("Protected-Non Protected in Forget Set")

        plt.suptitle(target_protected)
        plt.tight_layout()
        plt.savefig(savepath)
