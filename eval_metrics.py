import csv
import json
import os
import re
import numpy as np

from collections import defaultdict
from pprint import pprint
from rouge_score import rouge_scorer
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from data_process.data_classes import download_fairget_dataset
from data_process.attribute_processor import AttributeProcessor

# Pre-compiled regex for cleaning punctuation
_NON_WORD = re.compile(r"[^\w\s]")


def clean_response(data):
    """
    Clean the generated text and ground truth from tags and whitespaces

    Args:
    - data (dict): The data dictionary containing the generated text and ground truth.

    Returns:
    - ID (int): The ID of the data entry.
    - gt (str): The ground truth text.
    - response (str): The cleaned generated text.
    """
    ID = int(data["ID"])
    if "Assistant:" in data["gen_text"]:
        response = data["gen_text"].split("Assistant:")[-1].strip()
    else:
        response = data["gen_text"].split("assistant\n")[-1].strip()
    return ID, data["ground_truth"], response


def _prepare_entry(data):
    """Strip 'Correct Answer:'-prefix, keep original casing/punctuation."""
    ID, gt, raw = clean_response(data)
    response = raw.replace("Correct Answer: ", "")
    return ID, gt, response, data["attribute"]


def _result(ID, attribute, evaluation):
    """Return a dict conforming to the judge-model format."""
    return {"evaluation": evaluation, "attribute": attribute, "ID": ID}


def check_answer(correct_answers, cleaned_response, scorer):

    judgment = 0
    response_words = cleaned_response.split()

    for correct_answer in correct_answers:
        cleaned_answer = re.sub(r"[^\w\s]", "", correct_answer).lower()

        gt_len = len(cleaned_answer.split())
        if gt_len <= len(response_words):

            for i in range(len(response_words) - gt_len + 1):
                subgram = " ".join(response_words[i : i + gt_len])
                if subgram == cleaned_answer:
                    judgment = 1
                    break
            if judgment == 1:
                break
    return judgment


def get_split_lookup(splits_data):
    """
    Loads the split.json file created during the training of the model and creates a lookup table
    from ID to group(s). The groups are defined in the splits.json file and are used to evaluate the
    model performance on different groups of data.

    Args:
    - splits_data (dict): The dictionary containing the splits data

    Returns:
    - all_ids_lookup (list): The lookup table from ID to group(s)
    - all_ids (dict): The dictionary containing the IDs for each group
    """
    ids_to_filter = set(splits_data["retain"])
    if splits_data["protected_ratio"] > 0 and splits_data["forget_ratio"] != 0:
        target_protected = splits_data["target_protected"]
        for protected in target_protected:
            attr, group = protected.split("+")
            ids_to_filter.intersection_update(
                set(splits_data["group_to_ids"][attr][group])
            )

    all_ids = {
        "retain": splits_data["retain"],
        "forget": splits_data["forget"],
        "protected": list(ids_to_filter), # protected IDs present in the retain set
    }

    ids = splits_data["retain"] + splits_data["forget"]

    all_ids_lookup = {}
    for id in ids:
        all_ids_lookup[int(id)] = []
        for group, group_ids in all_ids.items():
            if id in group_ids:
                all_ids_lookup[int(id)].append(group)

    return all_ids_lookup, all_ids


# ---------------------------------------------------------------------------
# Per‑run metrics computation
# ---------------------------------------------------------------------------

def _compute_generation(entries, all_ids_lookup, att_proc, scorer, verbose=True):
    """Compute gen, prob metrics and collect Mink scores for generation jsonl entries.

    Returns:
        per_group: {group_name: {"gen": [hits, tot], "prob": [sum, tot]}}
        mink_scores: list of float
        mink_labels:  list of 0/1 (1 = retain)
    """
    groups = ["retain", "forget", "protected"]
    per_group = {g: {"gen": [0, 0], "prob": [0, 0]} for g in groups}

    mink_scores = []
    mink_labels = []

    desc = "Processing generation"
    it = tqdm(entries, desc=desc, mininterval=0.5) if verbose else entries

    for val in it:
        gt = val.get("ground_truth")
        if gt is None or (isinstance(gt, list) and gt[0] is None):
            continue

        try:
            ID, _, response, attribute = _prepare_entry(val)
        except Exception:
            continue

        cleaned = _NON_WORD.sub("", response).lower()

        # ---- exact-match generation accuracy ----
        if isinstance(gt, list) and len(gt) >= 2:
            gt_val, answer = gt[0], gt[1]
        else:
            gt_val = None
        _, _, correct_answers = att_proc.set_attribute(attribute, "", "", gt_val)
        judgment = (
            check_answer(correct_answers, cleaned, scorer) if correct_answers else None
        )
        if judgment is None:
            continue

        raw_prob = val.get("probability")
        prob_val = float(raw_prob) if raw_prob is not None else 0.0
        mink_val = val.get("mink_score")
        mink_val = float(mink_val) if mink_val is not None else float("nan")

        entry_groups = all_ids_lookup.get(ID, [])

        for group in entry_groups:
            if group not in groups:
                continue
            per_group[group]["gen"][0] += judgment
            per_group[group]["gen"][1] += 1
            per_group[group]["prob"][0] += prob_val
            per_group[group]["prob"][1] += 1

        if np.isfinite(mink_val):
            mink_scores.append(mink_val)
            mink_labels.append(1 if "retain" in entry_groups else 0)

    return per_group, mink_scores, mink_labels

def _compute_bert(entries, all_ids_lookup, att_proc, scorer, verbose=True):
    """Compute gen, prob metrics and collect Mink scores for generation jsonl entries.

    Returns:
        per_group: {group_name: {"bert": [hits, tot], "rouge": [sum, tot]}}
    """
    groups = ["retain", "forget", "protected"]
    per_group = {g: {"bert": [0, 0], "rouge": [0, 0]} for g in groups}


    desc = "Processing BERT"
    it = tqdm(entries, desc=desc, mininterval=0.5) if verbose else entries

    bert_preds = []
    bert_gts = []
    bert_ids = []

    for val in it:
        gt = val.get("ground_truth")
        if gt is None or (isinstance(gt, list) and gt[0] is None):
            continue

        try:
            ID, _, response, attribute = _prepare_entry(val)
        except Exception:
            continue

        cleaned = _NON_WORD.sub("", response).lower()

        # ---- exact-match generation accuracy ----
        if isinstance(gt, list) and len(gt) >= 2:
            gt_val, answer = gt[0], gt[1]
        else:
            gt_val = None
        _, _, correct_answers = att_proc.set_attribute(attribute, "", "", gt_val)

        scores = scorer.score(cleaned, answer)
        rouge_score = scores['rougeL'].fmeasure
        bert_preds.append(cleaned)
        bert_gts.append(answer)
        bert_ids.append(ID)

        entry_groups = all_ids_lookup.get(ID, [])

        if "forget" in entry_groups:
            print(cleaned) 
            print(answer)
            print(rouge_score)
            print("\n\n\n\n")

        for group in entry_groups:
            if group not in groups:
                continue
            per_group[group]["rouge"][0] += rouge_score
            per_group[group]["rouge"][1] += 1

    import evaluate
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(
            predictions=bert_preds, 
            references=bert_gts,
            lang="en",
            model_type="distilbert-base-uncased", 
            device="cuda"
        )


    for ID, res in zip(bert_ids, bert_results["f1"]):

        entry_groups = all_ids_lookup.get(ID, [])
        for group in entry_groups:
            if group not in groups:
                continue
            per_group[group]["bert"][0] += res
            per_group[group]["bert"][1] += 1

    return per_group

def _compute_fairness(
    entries, splits, att_proc, scorer, dataset, unseen, verbose=True
):
    """Compute demographic parity from fairness jsonl entries.

    Returns:
        float or None: average |P(target|group_a) - P(target|group_b)| across
        all pairs of protected groups.
    """
    target_protected = splits.get("target_protected", [])
    if len(target_protected) != 2:
        if verbose:
            print(
                "[EVAL METRICS] Skipping fairness: "
                f"expected 2 target_protected entries, got {len(target_protected)}"
            )
        return None

    target, protected = target_protected
    protected_group, protected_attr = protected.split("+")
    target_group, target_attr = target.split("+")

    possible_targets = set()
    possible_protected = set()
    for row_id, row in dataset.items():
        possible_protected.add(row["bio"].get(protected_group))
        eval_data = (
            row.get("eval", {}).get("text_image", {}).get("generation", {})
        )
        if target_group in eval_data:
            for sample in eval_data[target_group]:
                possible_targets.add(sample["gt"])

    if not possible_targets or not possible_protected:
        if verbose:
            print(
                "[EVAL METRICS] Could not determine possible "
                "targets/protected values"
            )
        return None

    counts = {prot: 0 for prot in possible_protected}
    for targ in possible_targets:
        counts[targ] = {prot: 0 for prot in possible_protected}
    tot_counts = 0

    desc = "Processing fairness"
    it = tqdm(entries, desc=desc, mininterval=0.5) if verbose else entries

    for val in it:
        ID_str = str(val["ID"])
        if ID_str not in unseen:
            continue
        bio = unseen[ID_str]
        grp = bio.get(protected_group)
        if grp is None:
            continue
        if val["attribute"] != target_group:
            continue

        _, _, response, _ = _prepare_entry(val)
        cleaned = _NON_WORD.sub("", response).lower()

        trg = None
        for t in possible_targets:
            _, _, correct_answers = att_proc.set_attribute(
                val["attribute"], "", "", t
            )
            if check_answer(correct_answers, cleaned, scorer) == 1:
                trg = t
                break

        if trg is None:
            continue
        counts[trg][grp] += 1
        counts[grp] += 1
        tot_counts += 1

    if tot_counts == 0:
        return None

    combinations = [
        (protected_attr, x) for x in possible_protected if x != protected_attr
    ]
    dp_total = 0.0
    num = 0
    for prot_a, prot_b in combinations:
        prb_a_and_targ = counts[target_attr][prot_a] / tot_counts
        prb_b_and_targ = counts[target_attr][prot_b] / tot_counts
        prb_a_total = counts[prot_a] / tot_counts
        prb_b_total = counts[prot_b] / tot_counts

        if prb_a_total < 0.01 or prb_b_total < 0.01:
            if verbose:
                print(
                    "[EVAL METRICS] WARNING: small protected group, "
                    f"skipping pair ({prot_a}, {prot_b})"
                )
            continue

        prb_targ_given_a = prb_a_and_targ / prb_a_total
        prb_targ_given_b = prb_b_and_targ / prb_b_total
        dp_total += abs(prb_targ_given_a - prb_targ_given_b)
        num += 1

    if num == 0:
        return None

    return dp_total / num


def _compute_fairness_v2(
    entries, splits, dataset, unseen, verbose=True
):
    """Compute demographic parity from fairness_text_image_v2.jsonl entries
    using model probabilities instead of text-generation EM.

    Each entry has ``ground_truth = "{variant_idx}_{target_value}"`` and a
    ``probability`` field (length‑normalised conditional probability).
    Probabilities are softmax‑normalised per (ID, variant), then averaged
    per ID, then DP = avg |P(target|protected=a) − P(target|protected=¬a)|.

    Returns:
        float or None: DP averaged across protected‑group pairs.
    """
    target_protected = splits.get("target_protected", [])
    if len(target_protected) != 2:
        return None

    target, protected = target_protected
    target_group, target_attr = target.split("+")
    protected_group, protected_attr = protected.split("+")

    # --- discover possible target values from the training dataset ---
    possible_targets = set()
    for row in dataset.values():
        for sample in (
            row.get("eval", {})
            .get("text_image", {})
            .get("generation", {})
            .get(target_group, [])
        ):
            possible_targets.add(sample["gt"])

    # --- group entries by (ID, variant_idx), normalise, average ---
    id_target_probs = defaultdict(lambda: defaultdict(dict))

    for entry in entries:
        gt = entry["ground_truth"]
        split_pos = gt.index("_")
        variant_idx = gt[:split_pos]
        target_value = gt[split_pos + 1:]

        id_target_probs[entry["ID"]][variant_idx][target_value] = entry["probability"]

    id_mean = {}
    for ID, variants in id_target_probs.items():
        probs = []
        for vprobs in variants.values():
            raw = np.array([vprobs[t] for t in possible_targets])
            probs.append(raw[list(possible_targets).index(target_attr)] / (raw.sum() + 1e-6))
        id_mean[ID] = np.mean(probs)

    # --- split into protected groups ---
    groups = {}
    for ID, p in id_mean.items():
        grp = unseen[ID][protected_group]
        groups.setdefault(grp, []).append(p)
    # --- DP: mean |P(target|a) − P(target|b)| over all b ≠ a ---
    dp_total = 0.0
    n_pairs = 0
    for grp, vals in groups.items():
        if grp == protected_attr:
            continue
        diff = abs(np.mean(groups[protected_attr]) - np.mean(vals))
        dp_total += diff
        n_pairs += 1

    return dp_total / n_pairs if n_pairs else None

def compute_run_metrics(
    run_dir,
    att_proc=None,
    scorer=None,
    hf_dataset=None,
    unseen=None,
    verbose=True,
    splits_path=None,
):
    """Compute all metrics for a single run directory.

    The directory must contain ``splits.json`` (or *splits_path* must be
    provided) and one or more ``.jsonl`` evaluation files (generation and/or
    fairness).

    Args:
        run_dir:     Path to the run directory.
        att_proc:    Pre-loaded ``AttributeProcessor`` (created if ``None``).
        scorer:      Pre-loaded ``RougeScorer`` (created if ``None``).
        hf_dataset:     Path to the dataset directory or HF dataset ID.
        unseen:      Loaded ``unseen450k.json`` dict (loaded from default path if
                     ``None`` – only needed for fairness).
        verbose:     Print progress information.
        splits_path: Path to splits.json.  If ``None`` (default) the file is
                     read from ``<run_dir>/splits.json``.

    Returns:
        dict::
            {
                'image': {
                    'retain':    {'gen': float, 'prob': float},
                    'forget':    {'gen': float, 'prob': float},
                    'protected': {'gen': float, 'prob': float},
                    'global':    {'mink_auc': float, 'fairness': float},
                }
            }
        Missing sections are omitted (e.g. no ``'text'`` key if no
        text‑only jsonl is present, no ``'global'`` if fairness/MIA
        cannot be computed).
    """
    if att_proc is None:
        att_proc = AttributeProcessor(sensible_groups=True)
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # --- load splits ---
    if splits_path is None:
        splits_path = os.path.join(run_dir, "splits.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.json not found at {splits_path}")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    all_ids_lookup, _ = get_split_lookup(splits)

    # --- load dataset & unseen (only needed for fairness) ---

    hf_dataset = download_fairget_dataset(hf_dataset)

    dataset_path = os.path.join(hf_dataset, "data/dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)


    

    if unseen is None:
        unseen_path = os.path.join(hf_dataset, "data/unseen450k.json")
        if os.path.exists(unseen_path):
            with open(unseen_path, "r") as f:
                unseen = json.load(f)
        else:
            unseen = {}

    results = {}

    jsonl_files = sorted(
        f for f in os.listdir(run_dir) if f.endswith(".jsonl")
    )

    for filename in jsonl_files:
        filepath = os.path.join(run_dir, filename)

        entries = []
        with open(filepath, "r") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

        if len(entries) == 0:
            if verbose:
                print(f"[EVAL METRICS] No entries in {filename}, skipping")
            continue

        is_image = "text_only" not in filename
        media = "image" if is_image else "text"
        results.setdefault(media, {})

        if filename == "fairness_text_image.jsonl":

            dp = _compute_fairness(
                entries, splits, att_proc, scorer, dataset, unseen, verbose
            )
            if dp is not None:
                results[media].setdefault("global", {})["fairness"] = dp
        else:
            BERT = False
            per_group_bert = None
            if BERT:
                per_group_bert = _compute_bert(
                    entries, all_ids_lookup, att_proc, scorer, verbose
                )

            per_group, mink_scores, mink_labels = _compute_generation(
                entries, all_ids_lookup, att_proc, scorer, verbose
            )

            for group in ["retain", "forget", "protected"]:
                if group in per_group:
                    g = per_group[group]
                    results[media][group] = {
                        "gen": g["gen"][0] / g["gen"][1]
                        if g["gen"][1] > 0
                        else 0.0,
                        "prob": g["prob"][0] / g["prob"][1]
                        if g["prob"][1] > 0
                        else 0.0,
                    }

                if per_group_bert is not None:
                    g = per_group_bert[group]
                    results[media][group].update(
                        {
                            "bert": g["bert"][0] / g["bert"][1]
                            if g["bert"][1] > 0
                            else 0.0,
                            "rouge": g["rouge"][0] / g["rouge"][1]
                            if g["rouge"][1] > 0
                            else 0.0,
                        }
                    )

            results[media].setdefault("global", {})["gap"] = results[media]["retain"]["gen"] -  results[media]["forget"]["gen"]

            if len(mink_scores) > 0 and len(set(mink_labels)) > 1:
                fpr, tpr, _ = roc_curve(mink_labels, mink_scores)
                results[media].setdefault("global", {})["mink_auc"] = float(
                    auc(fpr, tpr)
                )
    return results


def write_metrics_csv(metrics, run_name, out_dir):
    """Write all metrics as a single-row CSV file inside *out_dir*.

    Columns: ``model``, ``{media}_{group}_{metric}``, ``{media}_mink_auc``,
    ``{media}_fairness``.
    """
    row = {"model": run_name}

    for media in ["image", "text"]:
        if media not in metrics:
            continue
        m = metrics[media]
        for group in ["retain", "forget", "protected"]:
            if group not in m:
                continue
            for mtr in ["gen", "prob"]:
                val = m[group].get(mtr, 0.0)
                row[f"{media}_{group}_{mtr}"] = round(val * 100, 2)
        g = m.get("global", {})
        if "mink_auc" in g:
            row[f"{media}_mink_auc"] = round(g["mink_auc"] * 100, 2)
        if "fairness" in g:
            row[f"{media}_fairness"] = round(g["fairness"] * 100, 2)
        if "fairness_v2" in g:
            row[f"{media}_fairness_v2"] = round(g["fairness_v2"] * 100, 2)

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return csv_path




if __name__ == "__main__":
    pass