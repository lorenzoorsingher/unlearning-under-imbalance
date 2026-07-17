import random
import time
from operator import attrgetter

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from vectors_utils import ActivationExtractor, PersonaSteerer, compute_PCA_debias


def _all_gather_activations(activations, ids_tensor):
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return activations, ids_tensor

    world_size = dist.get_world_size()
    device = activations.device
    hidden = activations.shape[1]

    local_n = torch.tensor([activations.shape[0]], device=device, dtype=torch.long)
    all_n_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_n_list, local_n)
    all_n = [n.item() for n in all_n_list]
    max_n = max(all_n)

    if activations.shape[0] < max_n:
        pad = torch.zeros(max_n - activations.shape[0], hidden, device=device, dtype=activations.dtype)
        activations = torch.cat([activations, pad], dim=0)
        ids_pad = torch.full((max_n - ids_tensor.shape[0],), -1, device=device, dtype=ids_tensor.dtype)
        ids_tensor = torch.cat([ids_tensor, ids_pad], dim=0)

    gathered_acts = [torch.zeros_like(activations) for _ in range(world_size)]
    gathered_ids = [torch.zeros_like(ids_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_acts, activations)
    dist.all_gather(gathered_ids, ids_tensor)

    result_acts = torch.cat([gathered_acts[i][:all_n[i]] for i in range(world_size)], dim=0)
    result_ids = torch.cat([gathered_ids[i][:all_n[i]] for i in range(world_size)], dim=0)

    return result_acts, result_ids


def _sync_stop_flag(stop, accelerator):
    """Synchronize a stop/time-exceeded flag across all distributed ranks.

    When training with a time budget, different ranks may read slightly different
    local clock values from ``time.time()``.  If one rank decides to exit the
    training loop before the others, the remaining ranks will hang at the next
    NCCL collective (``accelerator.backward``) waiting for the exited rank to
    participate.

    This helper does an ``all_reduce(MAX)`` so that if **any** rank wants to
    stop, **all** ranks stop together, preventing NCCL watchdog timeouts.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return stop
    flag = torch.tensor([int(stop)], device=accelerator.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def persona_epoch7(
    dataloader_f,
    dataloader_r,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    mask=None,
    args=None,
    processor=None,
    # budget=-1,
    sweep_dataloader_f=None,
    sweep_dataloader_r=None,
    time_budget=None,
):

    target_layers = [int(x) for x in args.target_layers.split("-")]
    n_hidden_layers = model.module.config.text_config.num_hidden_layers
    layer_list = list(range(n_hidden_layers))

    # if len(target_layers) > 1: raise ValueError("so far only 1 layer is supported (extraction logic)")

    # online vectors extraction
    # if accelerator.is_main_process or True:
        
    id_act_dict = [{"forget":{}, "retain":{}} for _ in range(n_hidden_layers)]

    _sweep_f = sweep_dataloader_f if sweep_dataloader_f is not None else dataloader_f
    # if accelerator.is_main_process:
    #     breakpoint()

    avg_tkns = False
    num_forget_acts = 0
    with torch.no_grad():
        for step, data_f in enumerate(tqdm(_sweep_f)):
            batch_f, plaintext_f, gts_f, ids = data_f
            with ActivationExtractor(model, args.model_id, target_layers, detach=True, module=args.module) as act_dict:
                    _ = model(**batch_f)

            for target_layer in target_layers:
                acts = act_dict[target_layer]
                for labels, act_tkns, id in zip(batch_f.labels, acts, ids):
                    last_ass_tkn = (labels == -100).sum() - 1

                    if avg_tkns:
                        act = act_tkns[last_ass_tkn:].mean(dim=0)
                    else:
                        act = act_tkns[last_ass_tkn]

                    if id not in id_act_dict[target_layer]["forget"]:
                        id_act_dict[target_layer]["forget"][id] = []
                    id_act_dict[target_layer]["forget"][id].append(act.cpu())

            num_forget_acts = step

            # if step >= 20:
            #     break

    _sweep_r = sweep_dataloader_r if sweep_dataloader_r is not None else dataloader_r
    # breakpoint()
    with torch.no_grad():
        for step, data_r in enumerate(tqdm(_sweep_r)):
            batch_r, plaintext_r, gts_r, ids = data_r
            with ActivationExtractor(model, args.model_id, target_layers, detach=True, module=args.module) as act_dict:
                _ = model(**batch_r)

            for target_layer in target_layers:
                acts = act_dict[target_layer]
                for labels, act_tkns, id in zip(batch_r.labels, acts, ids):
                    last_ass_tkn = (labels == -100).sum() - 1

                    if avg_tkns:
                        act = act_tkns[last_ass_tkn:].mean(dim=0)
                    else:
                        act = act_tkns[last_ass_tkn]

                    if id not in id_act_dict[target_layer]["retain"]:
                        id_act_dict[target_layer]["retain"][id] = []
                    id_act_dict[target_layer]["retain"][id].append(act.cpu())

            if step >= num_forget_acts:
                break

        h_vects = [None for _ in range(n_hidden_layers)]

        for target_layer in target_layers:
            h_vects_forget = torch.stack([v for ID, lst in id_act_dict[target_layer]["forget"].items() for v in lst])
            h_vects_retain = torch.stack([v for ID, lst in id_act_dict[target_layer]["retain"].items() for v in lst])

            h_ids_forget = torch.tensor([int(ID) for ID, lst in id_act_dict[target_layer]["forget"].items() for v in lst])
            h_ids_retain = torch.tensor([int(ID) for ID, lst in id_act_dict[target_layer]["retain"].items() for v in lst])
                        
            h_vects[target_layer] = (h_vects_forget, h_vects_retain, h_ids_forget, h_ids_retain)
    #-------------------------------------------

    # Gather activations across all processes so each holds the full set
    dist_device = accelerator.device
    for target_layer in target_layers:
        hf, hr, ids_f, ids_r = h_vects[target_layer]
        hf = hf.to(dist_device)
        hr = hr.to(dist_device)
        ids_f = ids_f.to(dist_device)
        ids_r = ids_r.to(dist_device)

        gathered_f, gathered_ids_f = _all_gather_activations(hf, ids_f)
        gathered_r, gathered_ids_r = _all_gather_activations(hr, ids_r)

        h_vects[target_layer] = (
            gathered_f.cpu(),
            gathered_r.cpu(),
            gathered_ids_f.cpu(),
            gathered_ids_r.cpu(),
        )



    n_components = int(args.gamma)
    print("[DEBIAS] Informed PCA before steering")
    print("[DEBIAS] Number of components:", n_components)

    pca_projections = [None for _ in range(n_hidden_layers)]

    averaged_forget_idvec = [[] for _ in range(n_hidden_layers)]
    averaged_retain_idvec = [[] for _ in range(n_hidden_layers)]
    for layer in tqdm(target_layers):

        h_vects_forget, h_vects_retain, h_ids_forget, h_ids_retain = h_vects[layer]  # [n_forget_samples, hidden]

        forget_activations = h_vects_forget.float()
        retain_activations = h_vects_retain.float()

        users_vectors_forget, users_vectors_retain = compute_PCA_debias(
            forget_activations, retain_activations, n_components
        )
        

        unique_forget_ids = torch.unique(h_ids_forget)
        averaged_forget_vectors = []
        for user_id in unique_forget_ids:
            user_indices = torch.where(h_ids_forget == user_id)[0]
            user_vectors = users_vectors_forget[user_indices]
            averaged_forget_vectors.append(user_vectors.mean(dim=0))

        averaged_forget_idvec[layer] = torch.stack(averaged_forget_vectors)

        unique_retain_ids = torch.unique(h_ids_retain)
        averaged_retain_vectors = []
        for user_id in unique_retain_ids:
            user_indices = torch.where(h_ids_retain == user_id)[0]
            user_vectors = users_vectors_retain[user_indices]
            averaged_retain_vectors.append(user_vectors.mean(dim=0))

        averaged_retain_idvec[layer] = torch.stack(averaged_retain_vectors)
        # averaged_retain_idvec = torch.stack(averaged_retain_idvec)

    # average projections and base beforehand
    global_all_retain_idvec_t = [[] for _ in range(n_hidden_layers)]
    for layer in target_layers:
        global_all_retain_idvec_t[layer] = averaged_retain_idvec[layer].mean(dim=0)


    epoch_start = time.time()
    time_exceeded = False
    progress_bar = tqdm(total=len(dataloader_r), desc="Training")
    for step, data_r in enumerate(dataloader_r):

        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch_r, plaintext_r, gts_r, ids = data_r

        optimizer.zero_grad()

        # --- build unlearning vectors ---

        if "Qwen" in args.model_id:
            hidden_size = model.module.config.hidden_size
        elif "Idefics3" in args.model_id:
            hidden_size = model.module.config.text_config.hidden_size

        identities_vectors = [torch.zeros(args.batch_size, hidden_size) for _ in range(len(layer_list))]

        sampled_fuid_idx = random.randint(0, len(dataloader_f.dataset.train_ids) - 1)
        for i, uid in enumerate(ids):
            for layer in target_layers:
                # sample and average idvect from retain set (projected or original based on mode)
                avg_forget_h = averaged_forget_idvec[layer][sampled_fuid_idx]
                global_avg_h = global_all_retain_idvec_t[layer]

                identities_vectors[layer][i] = avg_forget_h - global_avg_h

        # --- Steered Model Forward Passes ---
        with PersonaSteerer(
            model,
            args.model_id,
            target_layers,
            coef=args.coef,
            ruv=identities_vectors,
            projection=pca_projections,
        ):
            outputs_s = model(**batch_r)

            loss = outputs_s.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        if args.debug > 0:
            return loss.item()

        accelerator.log(
            {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.6f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

    return loss.item()


def train_epoch(
    dataloader,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    mask=None,
    args=None,
    # budget=-1,
    processor=None,
    time_budget=None,
):

    epoch_start = time.time()
    time_exceeded = False
    progress_bar = tqdm(total=len(dataloader), desc="Training")

    for step, data in enumerate(dataloader):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch, plaintext, gts, ids = data

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if mask is not None:
            for name, param in model.named_parameters():
                if name in mask:
                    param.grad *= mask[name]

        optimizer.step()
        lr_scheduler.step()

        accelerator.log(
            {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
            step=step,
        )

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

    return loss.item()

def finetune_epoch(
    dataloader_t,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    args=None,
    processor=None,
):

    progress_bar = tqdm(total=len(dataloader_t), desc="Training")

    for step, data in enumerate(dataloader_t):

        batch, plaintext, gts, ids = data

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

        accelerator.log(
            {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],

            },
            step=step,
        )

    return loss.item()


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss


def SimNPO_epoch(
    dataloader_f,
    dataloader_r,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    args=None,
    # budget=-1,
    processor=None,
    time_budget=None,
):

    epoch_start = time.time()
    time_exceeded = False
    total_loss = 0
    progress_bar = tqdm(total=len(dataloader_f), desc="NPO Training")

    retain_iter = iter(dataloader_r)

    for step, data in enumerate(dataloader_f):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch, plaintext, gts, ids = data

        try:
            data_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(dataloader_r)
            data_r = next(retain_iter)
        batch_r, plaintext_r, gts_r, ids_r = data_r

        optimizer.zero_grad()

        # student forward pass

        loss_mask = batch["labels"] != -100
        forget_outputs = model(**batch)
        forget_batch_loss = get_batch_loss(forget_outputs.logits, batch["labels"])
        forget_loss = forget_batch_loss / loss_mask.sum(-1)
        forget_loss = -F.logsigmoid(args.beta * forget_loss).mean() * 2 / args.beta

        retain_outputs = model(**batch_r)
        retain_loss = retain_outputs.loss

        loss = args.alpha * forget_loss + retain_loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        accelerator.log(
            {
                "loss": loss.item(),
                "forget_loss": forget_loss.item(),
                "retain_loss": retain_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

    return loss.item()


def gad_interleaved_epoch(
    dataloader_f,
    dataloader_r,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    args=None,
    time_budget=None,
    processor=None,
):

    epoch_start = time.time()
    time_exceeded = False
    retain_iter = iter(dataloader_r)
    progress_bar = tqdm(total=len(dataloader_f), desc="GAD Interleaved")

    for step, data_f in enumerate(dataloader_f):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch_f, plaintext_f, gts_f, ids_f = data_f

        try:
            data_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(dataloader_r)
            data_r = next(retain_iter)
        batch_r, plaintext_r, gts_r, ids_r = data_r

        optimizer.zero_grad()

        outputs_f = model(**batch_f)
        loss_f = -outputs_f.loss

        outputs_r = model(**batch_r)
        loss_r = outputs_r.loss

        loss = loss_f + loss_r

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        accelerator.log(
            {
                "loss": loss.item(),
                "forget_loss": loss_f.item(),
                "retain_loss": loss_r.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

    return loss.item()


def rlpo_interleaved_epoch(
    dataloader_f,
    dataloader_r,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    args=None,
    time_budget=None,
    processor=None,
):

    epoch_start = time.time()
    time_exceeded = False
    retain_iter = iter(dataloader_r)
    progress_bar = tqdm(total=len(dataloader_f), desc="RL/PO Interleaved")

    for step, data_f in enumerate(dataloader_f):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch_f, plaintext_f, gts_f, ids_f = data_f

        try:
            data_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(dataloader_r)
            data_r = next(retain_iter)
        batch_r, plaintext_r, gts_r, ids_r = data_r

        optimizer.zero_grad()

        outputs_f = model(**batch_f)
        loss_f = outputs_f.loss

        outputs_r = model(**batch_r)
        loss_r = outputs_r.loss

        loss = loss_f + loss_r

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        accelerator.log(
            {
                "loss": loss.item(),
                "forget_loss": loss_f.item(),
                "retain_loss": loss_r.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

    return loss.item()


def GA_epoch(
    dataloader,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    mask=None,
    args=None,
    # budget=-1,
    processor=None,
    time_budget=None,
):

    epoch_start = time.time()
    time_exceeded = False
    total_loss = 0
    progress_bar = tqdm(total=len(dataloader), desc="GA Training")

    for step, data in enumerate(dataloader):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch, plaintext, gts, ids = data

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = -outputs.loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if mask is not None:
            for name, param in model.named_parameters():
                if name in mask:
                    param.grad *= mask[name]

        optimizer.step()
        lr_scheduler.step()

        accelerator.log(
            {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
            step=step,
        )

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

    return loss.item()


_LAYER_PATHS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "model.language_model.layers",
    "HuggingFaceM4/Idefics3-8B-Llama3": "base_model.model.model.text_model.layers",
}

def _get_layers(model, model_id):

    layers_path = _LAYER_PATHS[model_id]
    try:
        return attrgetter(layers_path)(model.module)
    except AttributeError:
        return attrgetter(layers_path)(model)


class EstimatedNet(nn.Module):
    def __init__(self, in_features, out_features, original_weight):
        super().__init__()
        self.down_proj = nn.Linear(in_features, out_features, bias=False)
        self.down_proj.weight.data.copy_(original_weight)

    def forward(self, x):
        return self.down_proj(x)


class _LunarCollector:
    def __init__(self, layers, target_layers):
        self.layers = layers
        self.target_layers = target_layers
        self.post_block = {i: [] for i in target_layers}
        self.pre_attn_ln = {i: [] for i in target_layers}
        self.pre_down_proj = {i: [] for i in target_layers}
        self.hooks = []

    def _post_block_hook(self, layer_idx):
        def hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self.post_block[layer_idx].append(act.detach().cpu())
        return hook

    def _pre_attn_ln_hook(self, layer_idx):
        def hook(module, input):
            self.pre_attn_ln[layer_idx].append(input[0].detach().cpu())
        return hook

    def _pre_down_proj_hook(self, layer_idx):
        def hook(module, input):
            self.pre_down_proj[layer_idx].append(input[0].detach().cpu())
        return hook

    def __enter__(self):
        for idx in self.target_layers:
            self.hooks.append(
                self.layers[idx].register_forward_hook(self._post_block_hook(idx))
            )
            self.hooks.append(
                self.layers[idx].post_attention_layernorm.register_forward_pre_hook(
                    self._pre_attn_ln_hook(idx)
                )
            )
            self.hooks.append(
                self.layers[idx].mlp.down_proj.register_forward_pre_hook(
                    self._pre_down_proj_hook(idx)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()

    def clear(self):
        for idx in self.target_layers:
            self.post_block[idx] = []
            self.pre_attn_ln[idx] = []
            self.pre_down_proj[idx] = []


def lunar_epoch(
    dataloader_f,
    dataloader_r,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    mask=None,
    args=None,
    processor=None,
    # budget=-1,
    sweep_dataloader_f=None,
    sweep_dataloader_r=None,
    time_budget=None,
):

    target_layers = [int(x) for x in args.target_layers.split("-")]

    lunar_coeff = getattr(args, "coef", 2.0)
    lunar_lr = getattr(args, "lunar_lr", 0.01)
    lunar_num_epochs = getattr(args, "lunar_num_epochs", getattr(args, "num_epochs", 10))
    lunar_reg = getattr(args, "lunar_reg", 0.1)

    accelerator.print(f"[LUNAR] target_layers={target_layers}, coef={lunar_coeff}, lr={lunar_lr}, epochs={lunar_num_epochs}, reg={lunar_reg}")

    # --- Phase A: Collect activations ---
    accelerator.print("[LUNAR] Phase A: Collecting activations from forget and retain sets...")


    layers = _get_layers(model, args.model_id)

    post_block_f_list = {i: [] for i in target_layers}
    pre_attn_ln_f_list = {i: [] for i in target_layers}
    pre_down_proj_f_list = {i: [] for i in target_layers}
    post_block_r_list = {i: [] for i in target_layers}
    pre_attn_ln_r_list = {i: [] for i in target_layers}
    pre_down_proj_r_list = {i: [] for i in target_layers}

    _sweep_f = sweep_dataloader_f 
    _sweep_r = sweep_dataloader_r 

    collector_f = _LunarCollector(layers, target_layers)
    collector_r = _LunarCollector(layers, target_layers)

    with torch.no_grad():
        with collector_f:
            for _batch_idx, data_f in enumerate(tqdm(_sweep_f, desc="LUNAR collect forget")):
                batch_f, _, _, _ = data_f
                _ = model(**batch_f)
                labels = batch_f["labels"]
                for i in range(labels.shape[0]):
                    last_tkn = (labels[i] == -100).sum().item() - 1
                    if last_tkn < 0:
                        continue
                    for lyr in target_layers:
                        post_block_f_list[lyr].append(
                            collector_f.post_block[lyr][0][i, last_tkn, :].cpu()
                        )
                        pre_attn_ln_f_list[lyr].append(
                            collector_f.pre_attn_ln[lyr][0][i, last_tkn, :].cpu()
                        )
                        pre_down_proj_f_list[lyr].append(
                            collector_f.pre_down_proj[lyr][0][i, last_tkn, :].cpu()
                        )
                for lyr in target_layers:
                    collector_f.post_block[lyr].clear()
                    collector_f.pre_attn_ln[lyr].clear()
                    collector_f.pre_down_proj[lyr].clear()

                
        collector_f = None
        n_forget_samples = len(post_block_f_list[target_layers[0]])

        with collector_r:
            for _batch_idx, data_r in enumerate(tqdm(_sweep_r, desc="LUNAR collect retain")):

                batch_r, _, _, _ = data_r
                _ = model(**batch_r)
                labels = batch_r["labels"]
                for i in range(labels.shape[0]):
                    last_tkn = (labels[i] == -100).sum().item() - 1
                    if last_tkn < 0:
                        continue
                    for lyr in target_layers:
                        post_block_r_list[lyr].append(
                            collector_r.post_block[lyr][0][i, last_tkn, :].cpu()
                        )
                        pre_attn_ln_r_list[lyr].append(
                            collector_r.pre_attn_ln[lyr][0][i, last_tkn, :].cpu()
                        )
                        pre_down_proj_r_list[lyr].append(
                            collector_r.pre_down_proj[lyr][0][i, last_tkn, :].cpu()
                        )
                for lyr in target_layers:
                    collector_r.post_block[lyr].clear()
                    collector_r.pre_attn_ln[lyr].clear()
                    collector_r.pre_down_proj[lyr].clear()
                if len(post_block_r_list[target_layers[0]]) >= n_forget_samples:
                    break
        collector_r = None

    for lyr in target_layers:
        nf = len(post_block_f_list[lyr])
        nr = len(post_block_r_list[lyr])
        accelerator.print(f"[LUNAR] Layer {lyr}: collected {nf} forget samples, {nr} retain samples")


    accelerator.print("[LUNAR] Phase B: Training EstimatedNet...")

    acc_device = accelerator.device
    trained_weights = {}
    final_loss = 0.0

    for lyr in target_layers:
        accelerator.print(f"[LUNAR] Training EstimatedNet for layer {lyr}...")

        post_block_f = torch.stack(post_block_f_list[lyr]).float().to(acc_device)
        pre_attn_ln_f = torch.stack(pre_attn_ln_f_list[lyr]).float().to(acc_device)
        pre_down_proj_f = torch.stack(pre_down_proj_f_list[lyr]).float().to(acc_device)
        post_block_r = torch.stack(post_block_r_list[lyr]).float().to(acc_device)
        pre_attn_ln_r = torch.stack(pre_attn_ln_r_list[lyr]).float().to(acc_device)
        pre_down_proj_r = torch.stack(pre_down_proj_r_list[lyr]).float().to(acc_device)

        mean_post_block_r = post_block_r.mean(dim=0)
        mean_post_block_f = post_block_f.mean(dim=0)
        direction = mean_post_block_r - mean_post_block_f

        perturbed_post_block_f = post_block_f + lunar_coeff * direction

        target_f = perturbed_post_block_f - pre_attn_ln_f

        lora_module = layers[lyr].mlp.down_proj
        is_peft_lora = hasattr(lora_module, "lora_A") and hasattr(lora_module, "base_layer")

        if is_peft_lora:
            base_weight = lora_module.base_layer.weight.data.float()
            delta = (lora_module.lora_B["default"].weight.data.float() @
                     lora_module.lora_A["default"].weight.data.float()) * lora_module.scaling["default"]
            effective_weight = base_weight + delta
        else:
            effective_weight = lora_module.weight.data.float().clone()

        in_features = lora_module.base_layer.in_features if is_peft_lora else lora_module.in_features
        out_features = lora_module.base_layer.out_features if is_peft_lora else lora_module.out_features

        est_net = EstimatedNet(in_features, out_features, effective_weight).to(acc_device)
        est_net.train()

        dataset = torch.utils.data.TensorDataset(pre_down_proj_f, target_f)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

        est_optimizer = torch.optim.AdamW(est_net.parameters(), lr=lunar_lr)
        est_scheduler = torch.optim.lr_scheduler.ExponentialLR(est_optimizer, gamma=0.9)

        for epoch in tqdm(range(lunar_num_epochs), desc="LUNAR EstimatedNet training"):
            epoch_loss = 0.0
            n_batches = 0
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(acc_device)
                batch_targets = batch_targets.to(acc_device)
                est_optimizer.zero_grad()
                preds = est_net(batch_inputs)
                loss_f = F.mse_loss(preds, batch_targets)
                loss_reg = F.mse_loss(est_net.down_proj.weight, effective_weight)
                loss = loss_f + lunar_reg * loss_reg
                loss.backward()
                est_optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            est_scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            accelerator.print(f"[LUNAR] Layer {lyr} epoch {epoch+1}/{lunar_num_epochs} MSE: {avg_loss:.6f}")
            final_loss = avg_loss

        trained_weights[lyr] = est_net.down_proj.weight.data.clone().to(torch.bfloat16)

        del est_net, est_optimizer, est_scheduler, dataset, train_loader
        del post_block_f, pre_attn_ln_f, pre_down_proj_f
        del post_block_r, pre_attn_ln_r, pre_down_proj_r
        torch.cuda.empty_cache()

    # --- Phase C: Apply trained weights ---
    accelerator.print("[LUNAR] Phase C: Applying trained weights...")

    for lyr in target_layers:
        lora_module = layers[lyr].mlp.down_proj
        is_peft_lora = hasattr(lora_module, "lora_A") and hasattr(lora_module, "base_layer")

        if is_peft_lora:
            lora_module.base_layer.weight.data.copy_(trained_weights[lyr])
            lora_module.lora_A["default"].weight.data.zero_()
            lora_module.lora_B["default"].weight.data.zero_()
        else:
            lora_module.weight.data.copy_(trained_weights[lyr])

        accelerator.print(f"[LUNAR] Applied trained weight to layer {lyr} down_proj")

    accelerator.print(f"[LUNAR] Complete. Final MSE loss: {final_loss:.6f}")
    return final_loss


LUNAR_RETAIN_STEPS_PER_ITER = 200


def iterative_lunar_epoch(
    sweep_forget_dl,
    sweep_retain_dl,
    retain_dl,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    args,
    processor,
    time_budget,
):
    target_layers = [int(x) for x in args.target_layers.split("-")]
    lunar_coeff = getattr(args, "coef", 2.0)
    lunar_lr = getattr(args, "lunar_lr", 0.01)
    lunar_num_epochs = getattr(args, "lunar_num_epochs", getattr(args, "num_epochs", 10))
    lunar_reg = getattr(args, "lunar_reg", 0.1)

    accelerator.print(f"[LUNAR-Iter] target_layers={target_layers}, coef={lunar_coeff}, lr={lunar_lr}, est_epochs={lunar_num_epochs}, reg={lunar_reg}")
    accelerator.print(f"[LUNAR-Iter] retain steps per iteration: {LUNAR_RETAIN_STEPS_PER_ITER}")

    layers = _get_layers(model, args.model_id)

    # === Phase A: Collect activations ONCE ===
    accelerator.print("[LUNAR-Iter] Phase A: Collecting activations once...")

    post_block_f_list = {i: [] for i in target_layers}
    pre_attn_ln_f_list = {i: [] for i in target_layers}
    pre_down_proj_f_list = {i: [] for i in target_layers}
    post_block_r_list = {i: [] for i in target_layers}

    collector_f = _LunarCollector(layers, target_layers)
    collector_r = _LunarCollector(layers, target_layers)

    with torch.no_grad():
        with collector_f:
            for _batch_idx, data_f in enumerate(tqdm(sweep_forget_dl, desc="LUNAR-Iter collect forget")):
                batch_f, _, _, _ = data_f
                _ = model(**batch_f)
                labels = batch_f["labels"]
                for i in range(labels.shape[0]):
                    last_tkn = (labels[i] == -100).sum().item() - 1
                    if last_tkn < 0:
                        continue
                    for lyr in target_layers:
                        post_block_f_list[lyr].append(
                            collector_f.post_block[lyr][0][i, last_tkn, :].cpu()
                        )
                        pre_attn_ln_f_list[lyr].append(
                            collector_f.pre_attn_ln[lyr][0][i, last_tkn, :].cpu()
                        )
                        pre_down_proj_f_list[lyr].append(
                            collector_f.pre_down_proj[lyr][0][i, last_tkn, :].cpu()
                        )
                for lyr in target_layers:
                    collector_f.post_block[lyr].clear()
                    collector_f.pre_attn_ln[lyr].clear()
                    collector_f.pre_down_proj[lyr].clear()
        collector_f = None
        n_forget_samples = len(post_block_f_list[target_layers[0]])

        with collector_r:
            for _batch_idx, data_r in enumerate(tqdm(sweep_retain_dl, desc="LUNAR-Iter collect retain")):
                batch_r, _, _, _ = data_r
                _ = model(**batch_r)
                labels = batch_r["labels"]
                for i in range(labels.shape[0]):
                    last_tkn = (labels[i] == -100).sum().item() - 1
                    if last_tkn < 0:
                        continue
                    for lyr in target_layers:
                        post_block_r_list[lyr].append(
                            collector_r.post_block[lyr][0][i, last_tkn, :].cpu()
                        )
                for lyr in target_layers:
                    collector_r.post_block[lyr].clear()
                    collector_r.pre_attn_ln[lyr].clear()
                    collector_r.pre_down_proj[lyr].clear()
                if len(post_block_r_list[target_layers[0]]) >= n_forget_samples:
                    break
        collector_r = None

    for lyr in target_layers:
        nf = len(post_block_f_list[lyr])
        nr = len(post_block_r_list[lyr])
        accelerator.print(f"[LUNAR-Iter] Layer {lyr}: collected {nf} forget samples, {nr} retain samples")

    # Pre-compute EstimatedNet training data per layer (frozen activations, same each iteration)
    layer_data = {}
    for lyr in target_layers:
        post_block_f = torch.stack(post_block_f_list[lyr]).float()
        pre_attn_ln_f = torch.stack(pre_attn_ln_f_list[lyr]).float()
        pre_down_proj_f = torch.stack(pre_down_proj_f_list[lyr]).float()
        post_block_r = torch.stack(post_block_r_list[lyr]).float()

        mean_post_block_r = post_block_r.mean(dim=0)
        mean_post_block_f = post_block_f.mean(dim=0)
        direction = mean_post_block_r - mean_post_block_f
        perturbed_post_block_f = post_block_f + lunar_coeff * direction

        target_f = perturbed_post_block_f - pre_attn_ln_f

        layer_data[lyr] = (pre_down_proj_f, target_f)

    # Free the raw activation lists — we only need the pre-computed tensors above
    del post_block_f_list, pre_attn_ln_f_list, pre_down_proj_f_list
    del post_block_r_list

    # === Iterative loop ===
    acc_device = accelerator.device
    epoch_start = time.time()
    retain_iter = iter(retain_dl)
    iteration = 0
    time_exceeded = False
    while True:
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        accelerator.print(f"[LUNAR-Iter] Iteration {iteration + 1} — training EstimatedNet per layer...")

        # Phase B: Train EstimatedNet for each layer from current effective weight
        trained_weights = {}
        for lyr in target_layers:
            lora_module = layers[lyr].mlp.down_proj
            is_peft_lora = hasattr(lora_module, "lora_A") and hasattr(lora_module, "base_layer")

            if is_peft_lora:
                base_weight = lora_module.base_layer.weight.data.float()
                delta = (lora_module.lora_B["default"].weight.data.float() @
                         lora_module.lora_A["default"].weight.data.float()) * lora_module.scaling["default"]
                effective_weight = base_weight + delta
            else:
                effective_weight = lora_module.weight.data.float().clone()

            in_features = lora_module.base_layer.in_features if is_peft_lora else lora_module.in_features
            out_features = lora_module.base_layer.out_features if is_peft_lora else lora_module.out_features

            inputs_f, targets_f = layer_data[lyr]
            inputs_f = inputs_f.to(acc_device)
            targets_f = targets_f.to(acc_device)

            est_net = EstimatedNet(in_features, out_features, effective_weight).to(acc_device)
            est_net.train()

            dataset = torch.utils.data.TensorDataset(inputs_f, targets_f)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            est_optimizer = torch.optim.AdamW(est_net.parameters(), lr=lunar_lr)
            est_scheduler = torch.optim.lr_scheduler.ExponentialLR(est_optimizer, gamma=0.9)
            for epoch in tqdm(range(lunar_num_epochs), desc="Training EstimatedNet"):
                for batch_inputs, batch_targets in train_loader:
                    batch_inputs = batch_inputs.to(acc_device)
                    batch_targets = batch_targets.to(acc_device)
                    est_optimizer.zero_grad()
                    preds = est_net(batch_inputs)
                    loss_f = F.mse_loss(preds, batch_targets)
                    loss_reg = F.mse_loss(est_net.down_proj.weight, effective_weight)
                    loss = loss_f + lunar_reg * loss_reg
                    loss.backward()
                    est_optimizer.step()
                est_scheduler.step()

            trained_weights[lyr] = est_net.down_proj.weight.data.clone().to(torch.bfloat16)

            del est_net, est_optimizer, est_scheduler, dataset, train_loader
            torch.cuda.empty_cache()

        # Phase C: Apply weights (overwrite base, keep LoRA — do NOT zero)
        for lyr in target_layers:
            lora_module = layers[lyr].mlp.down_proj
            is_peft_lora = hasattr(lora_module, "lora_A") and hasattr(lora_module, "base_layer")
            if is_peft_lora:
                lora_module.base_layer.weight.data.copy_(trained_weights[lyr])
            else:
                lora_module.weight.data.copy_(trained_weights[lyr])

        accelerator.print(f"[LUNAR-Iter] Weights applied — running {LUNAR_RETAIN_STEPS_PER_ITER} retain steps...")

        # Retain training: fixed number of steps
        for _ in range(LUNAR_RETAIN_STEPS_PER_ITER):
            try:
                data_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_dl)
                data_r = next(retain_iter)
            batch_r, _, _, _ = data_r

            optimizer.zero_grad()
            outputs_r = model(**batch_r)
            loss_r = outputs_r.loss
            accelerator.backward(loss_r)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

        iteration += 1
        accelerator.log(
            {"lunar_iteration": iteration, "lr": lr_scheduler.get_last_lr()[0]},
            step=iteration,
        )
        if accelerator.is_main_process:
            accelerator.print(f"[LUNAR-Iter] Iteration {iteration} complete")

    return 0.0


def miu_interleaved_epoch(
    dataloader_f,
    dataloader_r,
    train_dataloader,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    mi,
    mine_optimizers,
    groups_info,
    epoch,
    args,
    processor=None,
    time_budget=None,
):
    from miu_utils import get_groups, get_num_groups, gradient_clipping, tune_mine

    num_groups = get_num_groups(groups_info)
    mine_steps = getattr(args, "mine_steps", 100)

    tune_mine(model, mi, train_dataloader, groups_info, mine_optimizers,
              mine_steps, accelerator, model_id=args.model_id)

    mi_unlearning = mi["unlearning"]

    n_hidden_layers = model.module.config.text_config.num_hidden_layers
    last_layer = n_hidden_layers - 1

    epoch_start = time.time()
    time_exceeded = False
    retain_iter = iter(dataloader_r)
    progress_bar = tqdm(total=len(dataloader_f), desc="MIU Interleaved")

    for step, data_f in enumerate(dataloader_f):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)

        batch_f, plaintext_f, gts_f, ids_f = data_f

        try:
            data_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(dataloader_r)
            data_r = next(retain_iter)
        batch_r, plaintext_r, gts_r, ids_r = data_r

        optimizer.zero_grad()

        with ActivationExtractor(model, args.model_id, [last_layer], detach=False, module="residual") as act_dict:
            _ = model(**batch_f)

        acts = act_dict[last_layer]
        feats = []
        for labels, act_tkns in zip(batch_f["labels"], acts):
            last_ass_tkn = (labels == -100).sum() - 1
            feat = act_tkns[last_ass_tkn].float() if last_ass_tkn >= 0 else act_tkns.float().mean(dim=0)
            feats.append(feat)
        feats = torch.stack(feats)
        feats = F.normalize(feats, dim=-1)

        group = get_groups(dataloader_f.dataset, batch_f, ids_f, groups_info)
        group_tilde = torch.randint_like(group, num_groups)

        forget_loss = mi_unlearning(feats, group, group_tilde)

        outputs_r = model(**batch_r)
        retain_loss = outputs_r.loss

        loss = args.alpha * forget_loss + (1 - args.alpha) * retain_loss

        accelerator.backward(loss)
        gradient_clipping(model, mi_unlearning)

        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        accelerator.log(
            {
                "loss": loss.item(),
                "forget_loss": forget_loss.item(),
                "retain_loss": retain_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if args.debug > 0:
            return loss.item()

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f} | forget: {forget_loss.item():.4f} | retain: {retain_loss.item():.4f}"
            )

    return loss.item()


def miu_epoch(
    dataloader_f,
    dataloader_r,
    train_dataloader,
    model,
    optimizer,
    lr_scheduler,
    accelerator,
    mi,
    mine_optimizers,
    groups_info,
    epoch,
    args,
    processor=None,
    time_budget=None,
):
    from miu_utils import get_groups, get_num_groups, gradient_clipping, tune_mine

    num_groups = get_num_groups(groups_info)

    mine_steps = getattr(args, "mine_steps", 100)

    tune_mine(model, mi, train_dataloader, groups_info, mine_optimizers,
              mine_steps, accelerator, model_id=args.model_id)

    mi_unlearning = mi["unlearning"]

    forgetting_epochs = args.forgetting_epochs if args.forgetting_epochs != -1 else args.num_epochs
    n_hidden_layers = model.module.config.text_config.num_hidden_layers
    last_layer = n_hidden_layers - 1

    epoch_start = time.time()
    time_exceeded = False

    if epoch < forgetting_epochs:
        progress_bar = tqdm(total=len(dataloader_f), desc="MIU Forget", leave=False)
        for step, data in enumerate(dataloader_f):

            batch, plaintext, gts, ids = data

            optimizer.zero_grad()

            with ActivationExtractor(model, args.model_id, [last_layer], detach=False, module="residual") as act_dict:
                _ = model(**batch)

            acts = act_dict[last_layer]
            feats = []
            for labels, act_tkns in zip(batch["labels"], acts):
                last_ass_tkn = (labels == -100).sum() - 1
                feat = act_tkns[last_ass_tkn].float() if last_ass_tkn >= 0 else act_tkns.float().mean(dim=0)
                feats.append(feat)
            feats = torch.stack(feats)
            feats = F.normalize(feats, dim=-1)

            group = get_groups(dataloader_f.dataset, batch, ids, groups_info)
            group_tilde = torch.randint_like(group, num_groups)

            loss = mi_unlearning(feats, group, group_tilde)

            accelerator.backward(loss)
            gradient_clipping(model, mi_unlearning)

            optimizer.step()
            lr_scheduler.step()

            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"forget_loss: {loss.item():.4f}")

            if args.debug > 0:
                break

    total_loss = 0.0
    progress_bar = tqdm(total=len(dataloader_r), desc="MIU Retain", leave=False)
    for step, data in enumerate(dataloader_r):
        if time_exceeded:
            break
        if time_budget is not None and (time.time() - epoch_start) > time_budget:
            time_exceeded = True
        if time_budget is not None:
            time_exceeded = _sync_stop_flag(time_exceeded, accelerator)
        batch, plaintext, gts, ids = data

        optimizer.zero_grad()

        outputs_r = model(**batch)
        ce_loss = outputs_r.loss

        accelerator.backward(ce_loss)

        optimizer.step()
        lr_scheduler.step()


        accelerator.log(
            {
                "loss": ce_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"loss: {ce_loss.item():.4f} | lr: {lr_scheduler.get_last_lr()[0]:.7f}"
            )

        if args.debug > 0:
            break

    return total_loss

