import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pprint import pprint

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.append("../")
sys.path.append("../../")

from tqdm import tqdm

from data_process.data_utils import (
    idk_prompts,
)
from data_process.attribute_processor import AttributeProcessor


# def download_fairget_dataset(hf_dataset="argmaxxer/FAIRGET", cache_dir=None):
#     if os.path.isdir(hf_dataset):
#         return hf_dataset
#     from huggingface_hub import snapshot_download

#     return snapshot_download(
#         repo_id=hf_dataset, repo_type="dataset", cache_dir=cache_dir
#     )

import os
import shutil
import tarfile

from huggingface_hub import HfApi, snapshot_download


def upload_fairget_dataset(
    local_dir,
    repo_id="argmaxxer/FAIRGET",
):
    """
    Packages train_images/ and test_images/ into two separate tars,
    keeps everything else (dataset.json, csvs, etc.) as-is, and
    uploads the whole thing to the Hugging Face Hub as a dataset repo.

    Args:
    - local_dir (str): path to the dataset root (contains train_images/,
      test_images/, dataset.json, etc.)
    - repo_id (str): target HF dataset repo, e.g. 'argmaxxer/FAIRGET'
    - tmp_dir (str): where to build the tars before upload. Defaults to a
      'staging' folder next to local_dir.
    """

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(local_dir)), "fairget_staging")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. tar the two image folders separately
    for split in ["train_images", "test_images"]:
        split_path = os.path.join(local_dir, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"[ERROR] expected folder not found: {split_path}")

        tar_path = os.path.join(tmp_dir, f"{split}.tar")
        print(f"[INFO] compressing {split_path} -> {tar_path}")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(split_path, arcname=split)

    # 2. copy over everything else uncompressed (skip the two image folders
    #    and anything already in tmp_dir)
    skip = {"train_images", "test_images", os.path.basename(tmp_dir)}
    for entry in os.listdir(local_dir):
        if entry in skip:
            continue
        src = os.path.join(local_dir, entry)
        dst = os.path.join(tmp_dir, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # 3. upload everything in tmp_dir to the Hub
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print(f"[INFO] uploading {tmp_dir} -> {repo_id}")
    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"[INFO] done. repo: https://huggingface.co/datasets/{repo_id}")
    return repo_id


def download_fairget_dataset(hf_dataset="argmaxxer/FAIRGET", cache_dir=None):
    """
    Downloads the FAIRGET dataset repo (train_images.tar + test_images.tar +
    loose metadata files), extracts the two tars in place, and returns the
    local directory containing the fully-unpacked dataset
    (train_images/, test_images/, dataset.json, etc.).

    If hf_dataset is already a local directory, it's returned as-is.
    """

    if os.path.isdir(hf_dataset):
        return hf_dataset

    repo_dir = snapshot_download(
        repo_id=hf_dataset, repo_type="dataset", cache_dir=cache_dir
    )

    marker = os.path.join(repo_dir, ".extracted")
    if not os.path.exists(marker):
        for split in ["train_images", "test_images"]:
            tar_path = os.path.join(repo_dir, f"{split}.tar")
            if os.path.exists(tar_path):
                print(f"[INFO] extracting {tar_path}")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(repo_dir)
        open(marker, "w").close()

    return repo_dir


def _load_pil_image(img_path, target_size=None):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image at {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        height, width = image.shape[:2]
        target_width, target_height = target_size
        width_ratio = target_width / width
        height_ratio = target_height / height
        ratio = min(width_ratio, height_ratio)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(image)


class IDE_eval_Dataset(Dataset):

    def __init__(
        self,
        hf_dataset=None,
        target_size=None,
        task=None,
        split=None,
        media_type=None,
        train_ids=None,
        log=False,
        no_prompt=False,
    ):

        super().__init__()

        self.task = task
        self.target_size = target_size
        self.no_prompt = no_prompt

        if train_ids is not None:
            self.train_ids = [int(i) for i in train_ids]

        if hf_dataset is None:
            raise ValueError("[ERROR] please provide a path to the dataset")
        hf_dataset = download_fairget_dataset(hf_dataset)

        if media_type not in ["text_image", "text_only"]:
            raise ValueError(
                "[ERROR] media_type must be set to either 'text_image' or 'text_only'"
            )
        if split not in ["train", "test"]:
            raise ValueError(
                "[ERROR] split must be set to either 'train' or 'test'"
            )
        self.media_type = media_type
        self.split = split

        if split in ["train"]:
            images_path = os.path.join(hf_dataset, "train_images")
        elif split in ["test"]:
            images_path = os.path.join(hf_dataset, "test_images")

        json_path = os.path.join(hf_dataset, "data/dataset.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.processor = AttributeProcessor(sensible_groups=True)

        

        self.dataset = {
            "ID": [],
            "text": [],
            "gt": [],
            "attribute": [],
            "image": [],  # image ID or -1 for text-only samples
        }

        self.image_paths = {}
        build_question = self.build_generation_task

        # max questions per attribute. For each user every attribute has multiple questions phrased differently,
        # in order to keep the evaluation manageable we limit the number of questions per attribute to MAXQUESTIONS.


        MAXQUESTIONS = {
            "mask": 1,
            "generation": 1,
        }

        for idx, row in tqdm(
            self.data.items(), total=len(self.data), desc="Loading samples"
        ):

            ID = row["ID"]

            if train_ids is not None and int(ID) not in self.train_ids:
                continue

            id_subdir = os.path.join(images_path, ID)
            self.image_paths[ID] = [
                os.path.join(id_subdir, f)
                for f in os.listdir(id_subdir)]


            task_data = row["eval"][self.media_type][self.task]

            
            for attribute, questions in task_data.items():

                for idx, question_data in enumerate(questions):

                    if idx >= MAXQUESTIONS[self.task]:
                        break

                    attribute_desc = self.processor.get_attr_description(attribute)

                    conversation, gt = build_question(question_data, attribute_desc)

                    image = ID if self.media_type in ["text_image"] else -1

                    self.dataset["ID"].append(ID)
                    self.dataset["image"].append(image)
                    self.dataset["text"].append(conversation)
                    self.dataset["gt"].append(gt)
                    self.dataset["attribute"].append(attribute)

        if log:
            print(f"[INFO] Loaded {len(self.dataset['ID'])} evaluation samples")

    def __len__(self):

        return len(self.dataset["ID"])

    def __getitem__(self, idx: int):


        prompt, prompt_mia = self.dataset["text"][idx]
        ID = self.dataset["ID"][idx]
        gt = self.dataset["gt"][idx]
        attribute = self.dataset["attribute"][idx]

        image = (
            -1
            if self.dataset["image"][idx] == -1
            else _load_pil_image(random.choice(self.image_paths[ID]), self.target_size)
        )

        # breakpoint()
        return {
            "image": image,
            "prompt": prompt,
            "prompt_mia": prompt_mia,
            "gt": gt,
            "ID": ID,
            "attribute": attribute,
        }

    def resize_image(self, image):
        if self.target_size is not None:
            height, width = image.shape[:2]
            target_width, target_height = self.target_size

            width_ratio = target_width / width
            height_ratio = target_height / height

            ratio = min(width_ratio, height_ratio)

            new_width = int(width * ratio)
            new_height = int(height * ratio)

            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image  

    def build_generation_task(self, question_data, attribute):

        prompt = f"""Answer the following open question in one clear sentence using only the context provided and your existing knowledge. Respond with no extra commentary or explanation.\n"""

        sentence = deepcopy(question_data["question"])
        answer = deepcopy(question_data["answer"])
        gt = deepcopy(question_data["gt"])

        if attribute == "race":
            sentence = (
                f"Choose between white, black, asian or indian. {sentence.capitalize()}"
            )
        if attribute == "gender":
            sentence = f"Choose between male or female. {sentence.capitalize()}"
        if attribute == "age group":
            sentence = f"Choose between young, adult or older. {sentence.capitalize()}"
        if attribute == "skin tone":
            sentence = f"Choose between light, medium or dark. {sentence.capitalize()}"

        if not self.no_prompt:
            sentence = prompt + sentence

        if self.media_type in ["text_image"]:
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": sentence}],
                }
            ]
        elif self.media_type in ["text_only"]:
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": sentence}]}
            ]

        conversation_mia = deepcopy(conversation) + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

        return (conversation, conversation_mia), (gt, answer)


class IDE_train_Dataset(Dataset):


    def __init__(
        self,
        hf_dataset=None,
        target_size=None,
        train_ids=None,
        log=False,
        mode="train",
        fraction=1.0,
        samples=None,
        exclude_text_samples=False,
    ):

        super().__init__()

        self.target_size = target_size
        self.fraction = fraction
        self.samples = samples
        self.exclude_text_samples = exclude_text_samples

        self.train_ids = train_ids
        if train_ids is not None:
            self.train_ids = [int(i) for i in train_ids]

        if hf_dataset is None:
            raise ValueError("[ERROR] please provide a path to the dataset")
        hf_dataset = download_fairget_dataset(hf_dataset)

        if mode not in ["train", "random", "po"]:
            raise ValueError(
                "[ERROR] mode must be set to either 'train' or 'random' or 'po'"
            )

        PO = False
        if mode == "po":
            PO = True
            mode = "train"

        images_path = os.path.join(hf_dataset, "train_images")

        json_path = os.path.join(hf_dataset, "data/dataset.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.processor = AttributeProcessor(sensible_groups=True)

        attributes_to_exclude = []
        attributes_to_exclude = [
            attr[0] for attr in self.processor.implicit_attributes
        ]

        self.dataset = {
            "ID": [],
            "text": [],
            "words": [],  # spans for question and answer
            "gts": [],  # ground truth answers
            "image": [],  # image ID or -1 for text-only samples
        }
        self.image_paths = {}

        for idx, row in tqdm(
            self.data.items(), total=len(self.data), desc="Loading samples"
        ):

            ID = row["ID"]

            if train_ids is not None and int(ID) not in self.train_ids:
                continue

            id_subdir = os.path.join(images_path, ID)
            self.image_paths[ID] = [
                os.path.join(id_subdir, f)
                for f in os.listdir(id_subdir)]

            # we keep track of the image as the user ID and add the image to the conversation
            # only on demand

            train_data = row[mode]

            for media_type in ["text_image", "text_only"]:

                # TODO: use attribute for targeted attribute unlearning
                for attribute, samples in train_data[media_type].items():

                    if attribute in attributes_to_exclude:
                        continue

                    for sample in samples:
                        question = sample["q"]

                        # the PO flag is usend in policy optimization unlearning, the answer prompts are replaced with
                        # predefined prompts like "I don't have any information about this person." etc.
                        if PO:
                            answer = self.get_PO_prompt()
                            sample["a_words"] = [answer]
                        else:
                            answer = sample["a"]

                        content = [{"type": "text", "text": question}]
                        content = [{"type": "image"}] + content

                        user = {"role": "user", "content": content}

                        assistant = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        }

                        if media_type == "text_only":
                            
                            if not exclude_text_samples:
                                # text-only sample with dummy image
                                conversation = [user, assistant]
                                self.dataset["ID"].append(ID)
                                self.dataset["text"].append(conversation)
                                self.dataset["words"].append(
                                    {"q": sample["q_words"], "a": sample["a_words"]}
                                )
                                self.dataset["gts"].append(sample["gt"])
                                self.dataset["image"].append(
                                    -1
                                )  # collate will handle the dummy image

                            # text-only sample with image
                            conversation = [user, assistant]
                            self.dataset["ID"].append(ID)
                            self.dataset["text"].append(conversation)
                            self.dataset["words"].append(
                                {"q": sample["q_words"], "a": sample["a_words"]}
                            )
                            self.dataset["gts"].append(sample["gt"])
                            self.dataset["image"].append(ID)

                        elif media_type == "text_image":
                            # text-image sample with image
                            conversation = [user, assistant]
                            self.dataset["ID"].append(ID)
                            self.dataset["text"].append(conversation)
                            self.dataset["words"].append(
                                {"q": sample["q_words"], "a": sample["a_words"]}
                            )
                            self.dataset["gts"].append(sample["gt"])
                            self.dataset["image"].append(ID)

        if log:
            print(f"[INFO] Loaded {len(self.dataset['ID'])} training samples")

        self.sampled_idxs = range(len(self.dataset["ID"]))

        if self.samples is not None:
            population = list(self.sampled_idxs)
            if self.samples <= len(population):
                # sample without replacement
                self.sampled_idxs = random.sample(population, self.samples)
            else:
                # sample with replacement (allow duplicates) when requested samples > available
                self.sampled_idxs = random.choices(population, k=self.samples)
            if log:
                print(
                    f"[INFO] {len(self.sampled_idxs)} samples will be used for training (set by samples parameter)"
                )
        else:
            num_samples = int(len(self.dataset["ID"]) * self.fraction)
            self.sampled_idxs = random.sample(self.sampled_idxs, num_samples)
            if log:
                print(
                    f"[INFO] {len(self.sampled_idxs)} samples will be used for training ({self.fraction*100:.2f}% of the dataset)"
                )

    def __len__(self):

        return len(self.sampled_idxs)

    def __getitem__(self, idx: int):


        idx = self.sampled_idxs[idx]

        prompt = self.dataset["text"][idx]
        ID = self.dataset["ID"][idx]
        words = self.dataset["words"][idx]
        gt = self.dataset["gts"][idx]

        image = (
            -1 if self.dataset["image"][idx] == -1 else _load_pil_image(random.choice(self.image_paths[ID]), self.target_size)
        )

        return {
            "image": image,
            "prompt": prompt,
            "ID": ID,
            "words": words,
            "gt": gt,
        }

    def resize_image(self, image):
        if self.target_size is not None:
            height, width = image.shape[:2]
            target_width, target_height = self.target_size

            width_ratio = target_width / width
            height_ratio = target_height / height

            ratio = min(width_ratio, height_ratio)

            new_width = int(width * ratio)
            new_height = int(height * ratio)

            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image  

    def get_PO_prompt(self):
        return random.choice(idk_prompts)

class LUNAR_train_dataset(Dataset):

    def __init__(
        self,
        nsamples=5000,
        media_type="text_only",
        log=False,
        harmful_json_path="data/LUNAR/harmful.json",
    ):
        super().__init__()

        self.media_type = media_type

        with open(harmful_json_path, "r") as f:
            self.harmful_instructions = json.load(f)

        self.dataset = {
            "ID": [],
            "text": [],
            "words": [],
            "gts": [],
            "image": [],
        }

        for i in range(nsamples):
            instruction = random.choice(self.harmful_instructions)["instruction"]

            content = [{"type": "text", "text": instruction}]
            content = [{"type": "image"}] + content

            user = {"role": "user", "content": content}
            assistant = {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
            }
            conversation = [user, assistant]

            self.dataset["ID"].append(i)
            self.dataset["text"].append(conversation)
            self.dataset["words"].append({"q": [], "a": []})
            self.dataset["gts"].append("refusal")
            self.dataset["image"].append(-1)

        if log:
            print(f"[INFO] Generated {nsamples} LUNAR samples from {harmful_json_path} ({len(self.harmful_instructions)} instructions available)")

    def __len__(self):
        return len(self.dataset["ID"])

    def __getitem__(self, idx: int):
        prompt = self.dataset["text"][idx]
        ID = self.dataset["ID"][idx]
        words = self.dataset["words"][idx]
        gt = self.dataset["gts"][idx]

        image = self.dataset["image"][idx]

        return {
            "image": image,
            "prompt": prompt,
            "ID": ID,
            "words": words,
            "gt": gt,
        }


class IDE_fairness_Dataset(Dataset):

    def __init__(
        self,
        hf_dataset=None,
        target_size=None,
        target_attribute=None,
        protected_attribute=None,
        log=False,
    ):

        super().__init__()

        self.target_size = target_size
        self.media_type = "text_image"

        if hf_dataset is None:
            raise ValueError("[ERROR] please provide a path to the dataset")
        hf_dataset = download_fairget_dataset(hf_dataset)

        if target_attribute is None:
            raise ValueError(
                "[ERROR] please provide a target attribute for fairness evaluation"
            )

        images_path = os.path.join(hf_dataset, "test_images")

        json_path = os.path.join(hf_dataset, "data/dataset.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)

        templates_path = os.path.join(hf_dataset, "data/template_questions.json")
        with open(templates_path, "r") as file:
            self.template_questions = json.load(file)

        users_path = os.path.join(hf_dataset, "data/unseen450k.json")
        with open(users_path, "r") as f:
            self.unseen_users = json.load(f)

        self.processor = AttributeProcessor(sensible_groups=True)

        

        self.dataset = {
            "ID": [],
            "text": [],
            "gt": [],
            "attribute": [],
            "image": [],  # image ID or -1 for text-only samples
        }

        self.image_paths = {}

        if target_attribute == "all":
            targets = ["politics", "relationship", "height", "annualsalary"]
        else:
            targets = [target_attribute]

        target_dict = {}
        for target_attribute in targets:

            target_dict[target_attribute] = {
                "possible_questions": set(),
                "possible_answers": set(),
            }

            for ID, row in tqdm(
                self.data.items(), total=len(self.data), desc="Loading samples"
            ):

                if target_attribute in row["eval"]["text_image"]["generation"]:
                    for sample in row["eval"]["text_image"]["generation"][
                        target_attribute
                    ]:
                        # one special case...
                        if isinstance(sample["gt"], dict):
                            gt = sample["gt"]["bin"]
                        else:
                            gt = sample["gt"]

                        target_dict[target_attribute]["possible_answers"].add(gt)
                        target_dict[target_attribute]["possible_questions"].add(
                            sample["question"]
                        )

        for ID, bio in tqdm(self.unseen_users.items()):

            id_subdir = os.path.join(images_path, ID)
            self.image_paths[ID] = [
                os.path.join(id_subdir, f)
                for f in os.listdir(id_subdir)]

            for target_attribute in targets:

                possible_answers = target_dict[target_attribute]["possible_answers"]
                possible_questions = target_dict[target_attribute]["possible_questions"]

                for rnd_idx in random.sample(range(len(self.image_paths[ID])), 5):

                    img_path = self.image_paths[ID][rnd_idx]

                    question = random.choice(list(possible_questions))

                    content = [{"type": "image"}, {"type": "text", "text": question}]

                    user = {"role": "user", "content": content}
                    conversation = [user]

                    if protected_attribute == "all":
                        protected_gt = "any"
                    else:
                        protected_gt = bio[protected_attribute].lower()

                    self.dataset["ID"].append(ID)
                    self.dataset["image"].append(img_path)
                    self.dataset["text"].append(conversation)
                    self.dataset["gt"].append(protected_gt)
                    self.dataset["attribute"].append(target_attribute)

        if log:
            print(f"[INFO] Loaded {len(self.dataset['ID'])} fairness samples")

    def __len__(self):

        return len(self.dataset["ID"])

    def __getitem__(self, idx: int):


        prompt = self.dataset["text"][idx]
        ID = self.dataset["ID"][idx]
        gt = self.dataset["gt"][idx]
        attribute = self.dataset["attribute"][idx]
        image = _load_pil_image(self.dataset["image"][idx], self.target_size)
        return {
            "image": image,
            "prompt": prompt,
            "prompt_mia": None,
            "gt": gt,
            "ID": ID,
            "attribute": attribute,
        }

    def resize_image(self, image):
        if self.target_size is not None:
            height, width = image.shape[:2]
            target_width, target_height = self.target_size

            width_ratio = target_width / width
            height_ratio = target_height / height

            ratio = min(width_ratio, height_ratio)

            new_width = int(width * ratio)
            new_height = int(height * ratio)

            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image  

class VQA_train_Dataset(Dataset):

    def __init__(
        self,
        img_path="/leonardo_work/IscrC_CFMM/.cache/huggingface/hub/datasets--lmms-lab--VQAv2/snapshots/32665d35052eb4a6d4414851c3c829a72754915a/data/",
        vqa_path="data/VQAv2",
        nsamples=1000,
    ):

        super().__init__()

        self.nsamples = nsamples
        # Load a JSON file for additional data processing
        json_file_path = os.path.join(vqa_path, "fsvqa_original_val_questions.json")
        with open(json_file_path, "r") as json_file:
            vqa_q = json.load(json_file)["questions"]
        json_file_path = os.path.join(vqa_path, "fsvqa_original_val_annotations.json")
        with open(json_file_path, "r") as json_file:
            vqa_a = json.load(json_file)["annotations"]

        files = os.listdir(img_path)
        val = [f for f in files if f.startswith("validation")]

        # Load and merge all parquet files in val

        dataframes = []
        for file in tqdm(val, desc="Loading parquet files"):
            df = pd.read_parquet(os.path.join(img_path, file))
            dataframes.append(df)

        merged_data = pd.concat(dataframes, ignore_index=True)

        MAX_SAMPL = 10000

        sampled_idxs = random.sample(range(len(vqa_q)), min(MAX_SAMPL, nsamples))

        self.dataset = []
        self._vqa_arrays = {}
        for idx in tqdm(sampled_idxs, desc="Processing sampled indices"):
            question_sample = vqa_q[idx]
            answer_sample = vqa_a[idx]

            question = question_sample["question"]
            answer = answer_sample["answers"][0]["answer"]
            img_id = question_sample["img_id"]

            if img_id not in self._vqa_arrays:

                image_data = merged_data[merged_data["image_id"] == img_id][
                    "image"
                ].values[0]["bytes"]

                image = cv2.imdecode(
                    np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
                )
                self._vqa_arrays[img_id] = image

            self.dataset.append(
                {
                    "question": question,
                    "answer": answer,
                    # "image": image,
                    "img_id": img_id,
                }
            )

        print(f"[INFO] Loaded {len(self.dataset)} samples from VQAv2")

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx: int):

        idx = idx % len(self.dataset)

        sample = self.dataset[idx]

        question = sample["question"]
        answer = sample["answer"]
        img_id = sample["img_id"]
        raw = self._vqa_arrays[img_id]
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        target_size = 256
        if height > width:
            new_height = target_size
            new_width = int((width / height) * target_size)
        else:
            new_width = target_size
            new_height = int((height / width) * target_size)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)

        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }

        assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}

        conversation = [user, assistant]

        return {
            "prompt": conversation,
            "image": image,
            "ID": "VQA",
            "words": {"q": [], "a": []},
            "gt": None,
        }


def _get_assistant_delimiter_tokens(processor):
    """
    Return the token ids that mark the start of an assistant response
    by comparing a user-only conversation with and without generation prompt.
    """
    tokenizer = processor.tokenizer

    # A tiny user message that won't affect tokenization of the assistant header
    conv_user = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]


    # User turn only, no assistant header
    tokens_user_no_gen = processor.apply_chat_template(
        conv_user, add_generation_prompt=False, tokenize=True
    )

    # User turn + assistant header (add_generation_prompt=True)
    tokens_user_with_gen = processor.apply_chat_template(
        conv_user, add_generation_prompt=True, tokenize=True
    )
    # tokenizer.decode(tokens_user_with_gen)
    # The assistant delimiter is exactly the tokens appended by the generation prompt
    assistant_delim = tokens_user_with_gen[0][len(tokens_user_no_gen[0]) :]
    # breakpoint()
    return torch.tensor(assistant_delim)


def train_collate_fn_qwen_mixed(examples, processor, args=None, generation=False):
    """
    A data collator function for Qwen that processes the input text and images,
    and ensures the number of image tokens matches the number of images.
    """

    conversations = []
    all_images = []
    texts = []
    gts = []
    ids = []

    processor_model = type(processor).__name__

    add_generation_prompt = False
    if generation:
        add_generation_prompt = True
        processor.tokenizer.padding_side = "left"

    # Derive the assistant end-of-prompt tokens dynamically from the tokenizer
    end_of_prompt = _get_assistant_delimiter_tokens(processor)

    for sample in examples:
        conversation = sample["prompt"]
        image = sample["image"]
        ids.append(sample.get("ID", None))
        gts.append(sample.get("gt", None))

        if image == -1:
            # conversation comes with an image field even when dealing with text-only samples. This is due to
            # idefics not being able to handle mixed batches of text-only and text-image samples.

            image = Image.new("RGB", (32, 32), (0, 0, 0))
            all_images.append(image)

        else:
            all_images.append(image)

        # Qwen wants the image to be in the conversation, not in a separate list.
        if processor_model == "Qwen2_5_VLProcessor":
            conversation[0]["content"][0]["image"] = image

        conversations.append(conversation)

        # Convert the conversation into a text template
        text = processor.apply_chat_template(conversation, add_generation_prompt=add_generation_prompt, tokenize=False)   
        text = processor.apply_chat_template(sample["prompt"], add_generation_prompt=add_generation_prompt, tokenize=False)          

        texts.append(text)

    if processor_model == "Qwen2_5_VLProcessor":
        all_images, _ = process_vision_info(conversations)

    batch = processor(
        text=texts,
        images=all_images,
        padding=True,
        return_tensors="pt",
        max_length=280,
        truncation=True,
    )

    # masking everything that comes before the assistant response
    labels = batch["input_ids"].clone()

    end_of_prompt = end_of_prompt.to(device=labels.device)

    for batch_idx, label_seq in enumerate(labels):
        keep_mask = torch.zeros_like(label_seq, dtype=torch.bool)

        prompt_end = 0
        # looping in reverse to find the last occurrence of the end of prompt token sequence
        for i in range(len(label_seq) - len(end_of_prompt) + 1)[::-1]:
            if torch.equal(label_seq[i : i + len(end_of_prompt)], end_of_prompt):
                keep_mask[i + len(end_of_prompt) :] = True
                prompt_end = i
                break
        # Mask all tokens EXCEPT the ones we want to keep (either entire assistant response or target words only)
        labels[batch_idx][~keep_mask] = -100

    batch["labels"] = labels

    return (batch, texts, gts, ids)


def train_collate_fn_qwen_mixed_fttp(examples, processor, args=None, generation=False):
    """
    A data collator function for Qwen that processes the input text and images,
    and ensures the number of image tokens matches the number of images.
    """

    conversations = []
    all_images = []
    texts = []
    enc_texts = []
    gts = []
    ids = []

    processor_model = type(processor).__name__

    add_generation_prompt = False
    if generation:
        add_generation_prompt = True
        processor.tokenizer.padding_side = "left"

    # Derive the assistant end-of-prompt tokens dynamically from the tokenizer
    end_of_prompt = _get_assistant_delimiter_tokens(processor)

    for sample in examples:
        conversation = sample["prompt"]
        image = sample["image"]
        ids.append(sample.get("ID", None))
        gts.append(sample.get("gt", None))

        if image == -1:
            # conversation comes with an image field even when dealing with text-only samples. This is due to
            # idefics not being able to handle mixed batches of text-only and text-image samples.

            image = Image.new("RGB", (32, 32), (0, 0, 0))
            all_images.append([image])

        else:
            all_images.append(image)

        # Qwen wants the image to be in the conversation, not in a separate list.
        if processor_model == "Qwen2_5_VLProcessor":
            conversation[0]["content"][0]["image"] = image

        conversations.append(conversation)

        # Convert the conversation into a text template
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=add_generation_prompt, tokenize=False
        )

        texts.append(text)

        enc_words = []
        # find words

        maxlen = 10
        curlen = 1
        encoded = processor.tokenizer.encode(text)

        # a_words = sample["words"]["a"] + [" " + a for a in sample["words"]["a"]] + [a + " " for a in sample["words"]["a"]]

        for _ in range(maxlen):
            for i in range(len(encoded) - curlen):
                if (
                    processor.tokenizer.decode(encoded[i : i + curlen]).strip()
                    in sample["words"]["a"]
                ):
                    enc_words.append(encoded[i : i + curlen])
                if len(enc_words) >= 2:
                    break
            curlen += 1
        # print(len(enc_words), "->", len(sample["words"]["a"]))
        enc_texts.append(enc_words)

    if processor_model == "Qwen2_5_VLProcessor":
        all_images, _ = process_vision_info(conversations)

    batch = processor(
        text=texts,
        images=all_images,
        padding=True,
        return_tensors="pt",
        max_length=280,
        truncation=True,
    )

    # masking everything that comes before the assistant response
    labels = batch["input_ids"].clone()

    end_of_prompt = end_of_prompt.to(device=labels.device)

    labels_r = labels.clone()
    labels_f = labels.clone()
    for batch_idx, label_seq in enumerate(labels):
        keep_mask = torch.zeros_like(label_seq, dtype=torch.bool)

        prompt_end = 0
        # looping in reverse to find the last occurrence of the end of prompt token sequence
        for i in range(len(label_seq) - len(end_of_prompt) + 1)[::-1]:
            if torch.equal(label_seq[i : i + len(end_of_prompt)], end_of_prompt):
                keep_mask[i + len(end_of_prompt) :] = True
                prompt_end = i
                break

        # Mask all tokens EXCEPT the ones we want to keep (either entire assistant response or target words only)

        enc_words = enc_texts[batch_idx]
        hide_mask = torch.zeros_like(labels[batch_idx], dtype=torch.bool)
        for word in enc_words:
            word = torch.Tensor(word).int()
            for i in range(len(labels[batch_idx]) - len(word) + 1)[::-1]:
                if torch.equal(labels[batch_idx][i : i + len(word)], word):
                    hide_mask[i : i + len(word)] = True
                    # print(processor.tokenizer.decode(word))
                    # print(i, ":", )
        # labels[batch_idx][~keep_mask] = -100

        labels_r[batch_idx][~keep_mask | hide_mask] = -100
        labels_f[batch_idx][~keep_mask | ~hide_mask] = -100
        labels[batch_idx][~keep_mask] = -100

    batch["labels"] = labels
    batch["labels_r"] = labels_r
    batch["labels_f"] = labels_f
    # breakpoint()

    # processor.tokenizer.decode(labels_f.abs())
    # processor.tokenizer.decode(labels_r.abs())
    return (batch, texts, gts, ids)


def eval_collate_fn_qwen(examples, processor):

    IDs = []
    gts = []
    conversations = []
    conversations_mia = []
    attributes = []
    image_inputs = []
    image_inputs_mia = []
    metadatas = []

    processor_model = type(processor).__name__

    has_prompt = examples[0]["prompt"] is not None
    has_mia = examples[0]["prompt_mia"] is not None
    text_only = examples[0]["image"] == -1

    for sample in examples:
        if processor_model == "Qwen2_5_VLProcessor":
            if not text_only:
                # Qwen wants the image to be in the conversation, not in a separate list.
                if has_prompt:
                    sample["prompt"][0]["content"][0]["image"] = sample["image"]
                if has_mia:
                    sample["prompt_mia"][0]["content"][0]["image"] = sample["image"]

        conversations.append(sample["prompt"])
        conversations_mia.append(sample["prompt_mia"])
        IDs.append(sample["ID"])
        gts.append(sample["gt"])
        attributes.append(sample["attribute"])
        image_inputs.append(sample["image"])
        image_inputs_mia.append(sample["image"])
        metadatas.append(sample.get("metadata", None))

    if processor_model == "Qwen2_5_VLProcessor":
        if has_prompt:
            image_inputs, _ = process_vision_info(conversations)
        if has_mia:
            image_inputs_mia, _ = process_vision_info(conversations_mia)
    else:
        if text_only:
            # remove the dummy images for text-only samples
            if has_prompt:
                image_inputs = None
            if has_mia:
                image_inputs_mia = None

    inputs = None
    text = None
    if has_prompt:
        text = processor.apply_chat_template(conversations, add_generation_prompt=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        text = [None] * len(conversations)

    inputs_mia = None
    if has_mia:
        text_mia = processor.apply_chat_template(
            conversations_mia, add_generation_prompt=False
        )
        inputs_mia = processor(
            text=text_mia,
            images=image_inputs_mia,
            padding=True,
            return_tensors="pt",
        )

    return inputs, inputs_mia, text, gts, IDs, attributes, metadatas


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data Preprocessing for LLAVA")
    args = parser.parse_args()

    # args.model_id = "HuggingFaceM4/idefics2-8b"
    args.model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
    args.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    args.cache_path = args.model_id

    args.hf_dataset = "argmaxxer/FAIRGET"

    args.splits_path = "data/split_base_2k_small.json"

    ################################
    processor = AutoProcessor.from_pretrained(args.model_id, do_image_splitting=False)

    if "Qwen" in args.model_id:
        special_tokens = ["<image>", "<pad>"]
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        processor.tokenizer.additional_special_tokens = list(
            set(processor.tokenizer.additional_special_tokens or [])
            | set(special_tokens)
        )
        processor.tokenizer.pad_token = "<pad>"
    elif "Idefics" in args.model_id:
        pass

    with open(args.splits_path, "r") as f:
        splits = json.load(f)
    # ide_ids = [i for i in range(10)]
    # ide_ids = None
    ids_retain = splits["retain"]  # use the retain split for training
    ids_forget = splits["forget"]  # use the forget split for forgetting

    TESTIN = "train"

    # ===================== testing the FAIRNESS dataset ===================== #
    if TESTIN == "fairness":

        fair_dataset = IDE_fairness_Dataset(
            hf_dataset=args.hf_dataset,
            target_size=(256, 256),
            target_attribute="all",
            protected_attribute="all",
            log=True,
        )
        # breakpoint()
        # 20000
        # fair_dataloader = DataLoader(
        #     fair_dataset,
        #     batch_size=64,
        #     shuffle=True,
        #     num_workers=4,
        #     pin_memory=True,
        #     collate_fn=lambda x: eval_collate_fn_qwen(x, processor),
        # )

        for sample in fair_dataset:
            breakpoint()

            inputs, inputs_mia, text, gts, IDs, attributes, metadatas = eval_collate_fn_qwen([sample], processor)

            # processor.tokenizer.decode(inputs["input_ids"][0])
            # processor.tokenizer.decode(inputs_mia["input_ids"][0])
    # ===================== testing the EVAL dataset =================
    elif TESTIN == "eval":

        eval_dataset = IDE_eval_Dataset(
            hf_dataset=args.hf_dataset,
            task="generation",
            target_size=(256, 256),
            media_type="text_image",
            split="test",
            log=True,
        )

        # for sample in eval_dataset:
        #     # breakpoint()
        #     batch = eval_collate_fn_qwen([sample], processor)

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: eval_collate_fn_qwen(x, processor),
        )

        for sample in tqdm(eval_dataloader):
            # breakpoint()
            print(sample)
            pass

    elif (
        TESTIN == "train"
    ):  # ===================== testing the TRAIN dataset ===================== #

        train_dataset = IDE_train_Dataset(
            hf_dataset=args.hf_dataset,
            target_size=(256, 256),
            train_ids=ids_retain,
            mode="train",
            log=True,
        )

        sample = train_dataset[0]
        collate_fn = train_collate_fn_qwen_mixed
        # collate_fn = train_collate_fn_qwen_mixed_fttp

        # for sample in train_dataset:
        #     batch, text, _, _ = collate_fn([sample], processor)
        #     pass
        train_collate_fn_qwen_mixed([sample], processor)
        breakpoint()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, processor),
        )

        # breakpoint()
        maxl = 0
        for sample in train_dataloader:
            batch, text, _, _ = sample
            # maxl = max(maxl, batch.input_ids.shape[-1])
            # pprint(f"{maxl}\t{batch.input_ids.shape[-1]}")

            maxl += 1
            if maxl >= 10:
                breakpoint()

                processor.tokenizer.decode(batch["labels_r"][0].abs())

    elif TESTIN == "lunar":
        
        train_dataset = LUNAR_train_dataset(
            nsamples=100,
            media_type="text_only",
            log=True,
            harmful_json_path="LUNAR/dataset/splits/harmful.json",
        )

        sample = train_dataset[0]
        batch = train_collate_fn_qwen_mixed([sample], processor)


        batch, text, _, _ = train_collate_fn_qwen_mixed([sample], processor)
        breakpoint()

        # batch, text, _, _ = train_collate_fn_qwen_mixed([sample], processor)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: train_collate_fn_qwen_mixed(
                x, processor
            ),
        )

        for sample in train_dataloader:
            batch, text, _, _ = sample
            breakpoint()
        processor.tokenizer.decode(batch.input_ids[0].abs())

