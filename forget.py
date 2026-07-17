import itertools

import torch
from torch.utils.data import DataLoader
from data_process.data_classes import IDE_train_Dataset, LUNAR_train_dataset

from utils import save_checkpoint
from methods import (
    GA_epoch,
    SimNPO_epoch,
    gad_interleaved_epoch,
    lunar_epoch,
    miu_epoch,
    miu_interleaved_epoch,
    persona_epoch7,
    rlpo_interleaved_epoch,
    train_epoch,
)

# Budget mode: "samples" = old behavior (modulate dataset sizes), "time" = time cap + interleaved
BUDGET_METHOD = "time"


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def forget(
    args,
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    splits,
    target_size,
    collate_fn,
    processor,
):

    # some methods require different versions of the dataset (e.g. random labels or idk prompts)
    mode = "train"
    if args.method in ["RL", "SalUn"]:
        mode = "random"
    if args.method in ["PO"]:
        mode = "po"

    # splits sanity check
    assert set(splits["retain"]).isdisjoint(
        set(splits["forget"])
    ), "retain and forget sets must not intersect"

    # ------ Compute Budget and adjust steps accordingly ------

    forget_samples = None
    multimodal_forget_dataset = IDE_train_Dataset(
        args.hf_dataset,
        target_size=target_size,
        train_ids=splits["forget"],
        log=accelerator.is_main_process,
        mode=mode,
        samples=None,
        exclude_text_samples = True
    )

    forget_samples = len(multimodal_forget_dataset)


    if BUDGET_METHOD == "time":
        time_budget = forget_samples * args.seconds_per_forget_sample
        time_budget = max(time_budget, 600)
        retain_samples = None
        if accelerator.is_main_process:
            print(f"[INFO] Time budget: {time_budget:.1f}s for {forget_samples} forget samples ({args.seconds_per_forget_sample}s per sample)")

    

    if args.method in ["MIU", "MIU_INTERLEAVED"]:
        multimodal_train_dataset = IDE_train_Dataset(
            args.hf_dataset,
            target_size=target_size,
            train_ids=splits["forget"] + splits["retain"],

            log=accelerator.is_main_process,

            mode=mode,
            samples=forget_samples,
        )
        train_dataloader = DataLoader(
            multimodal_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, processor),
            num_workers=4 * accelerator.num_processes,
        )

        target, protected = splits["target_protected"]
        target_attr, target_attribute = target.split("+")
        protected_attr, protected_attribute = protected.split("+")
        all_ids = multimodal_train_dataset.data.keys()

        all_targets = set()
        all_protected = set()
        for id, val in multimodal_train_dataset.data.items():
            bio = val["bio"]

            if target_attr in bio:
                all_targets.add(bio[target_attr])
            if protected_attr in bio:
                all_protected.add(bio[protected_attr])

        comb = list(itertools.product(all_targets, all_protected))
        groups = sorted(["-".join(el) for el in comb])
        groups_map = {g: idx for idx, g in enumerate(groups)}
        attrs = (target_attr, protected_attr)
        groups_info = attrs, groups_map

    multimodal_retain_dataset = IDE_train_Dataset(
        args.hf_dataset,
        target_size=target_size,
        train_ids=splits["retain"],
        log=accelerator.is_main_process,
        mode="train",
        samples=retain_samples,
        exclude_text_samples = True
    )

    # budget = approx_n_steps

    forget_dataloader = DataLoader(
        multimodal_forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=4 * accelerator.num_processes,
    )
    retain_dataloader = DataLoader(
        multimodal_retain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=4 * accelerator.num_processes,
    )

    if args.method in ["PV7"]:
        sweep_forget_dataloader = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size * 1,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, processor),
            num_workers=4,
        )
        if args.method == "LUNAR":
            sweep_multimodal_retain_dataset = LUNAR_train_dataset(
                nsamples=len(multimodal_forget_dataset),
                media_type="text_only",
                log=accelerator.is_main_process,
            )
            sweep_retain_dataloader = DataLoader(
                sweep_multimodal_retain_dataset,
                batch_size=args.batch_size * 1,
                shuffle=False,
                collate_fn=lambda x: collate_fn(x, processor),
                num_workers=4,
            )
        else:
            sweep_retain_dataloader = DataLoader(
                multimodal_retain_dataset,
                batch_size=args.batch_size * 1,
                shuffle=False,
                collate_fn=lambda x: collate_fn(x, processor),
                num_workers=4,
            )

        sweep_forget_dataloader, sweep_retain_dataloader = accelerator.prepare(
            sweep_forget_dataloader, sweep_retain_dataloader
        )



    model, optimizer, retain_dataloader, forget_dataloader, lr_scheduler = (
        accelerator.prepare(
            model,
            optimizer,
            retain_dataloader,
            forget_dataloader,
            lr_scheduler,
        )
    )

    if accelerator.is_main_process:
        print("Forget len:", len(forget_dataloader))
        print("Retain len:", len(retain_dataloader))
        print("Num epochs:", args.num_epochs)

    if args.method in ["MIU", "MIU_INTERLEAVED"]:
        import builtins

        import torch.distributed as dist

        from miu_utils import MINE, MutualInformation, get_num_groups, tune_mine

        if dist.is_initialized() and dist.get_rank() != 0:
            builtins.print = lambda *args, **kwargs: None

        if "Qwen" in args.model_id:
            z_size = model.module.config.hidden_size
        elif "Idefics3" in args.model_id:
            z_size = model.module.config.text_config.hidden_size
        else:
            z_size = model.module.config.hidden_size
        c_size = get_num_groups(groups_info)

        mine_unlearning = MINE(z_size=z_size, c_size=c_size)
        mi_unlearning = MutualInformation(mine_unlearning)
        mine_unlearning_optim = torch.optim.SGD(
            mine_unlearning.parameters(), lr=args.mine_lr, momentum=0.9
        )

        mi_unlearning, mine_unlearning_optim = accelerator.prepare(
            mi_unlearning, mine_unlearning_optim
        )
        mi = {"unlearning": mi_unlearning}
        mine_optim = {"unlearning": mine_unlearning_optim}

    torch.cuda.empty_cache()
    for epoch in range(args.num_epochs):
        model.train()

        # Gradient Ascent
        if args.method in ["GA"]:
            loss = GA_epoch(
                dataloader=forget_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )
        # Gradient Difference
        elif args.method in ["GAD"]:
            loss = gad_interleaved_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )

        elif args.method in ["PO"]:
            loss = rlpo_interleaved_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )
        # Random Labeling
        elif args.method in ["RL"]:
            loss = rlpo_interleaved_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )
        if args.method in ["SimNPO"]:
            loss = SimNPO_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )
        # Persona Unlearning ===============================================0
        elif args.method in ["PV7"]:
            if not hasattr(args, 'target_attr') or args.target_attr is None:
                target, protected = splits["target_protected"]
                args.target_attr = target.split("+")[0]
                args.protected_attr = protected.split("+")[0]
            loss = persona_epoch7(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                sweep_dataloader_f=sweep_forget_dataloader,
                sweep_dataloader_r=sweep_retain_dataloader,
                time_budget=time_budget,
            )
        elif args.method in ["LUNAR"]:

            loss = lunar_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                sweep_dataloader_f=sweep_forget_dataloader,
                sweep_dataloader_r=sweep_retain_dataloader,
            )

                # loss = train_epoch(
                #     dataloader=retain_dataloader,
                #     model=model,
                #     optimizer=optimizer,
                #     lr_scheduler=lr_scheduler,
                #     accelerator=accelerator,
                #     args=args,
                #     processor=processor,
                # )
        elif args.method in ["MIU"]:
            loss = miu_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                train_dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                mi=mi,
                mine_optimizers=mine_optim,
                groups_info=groups_info,
                epoch=epoch,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )
        elif args.method in ["MIU_INTERLEAVED"]:
            loss = miu_interleaved_epoch(
                dataloader_f=forget_dataloader,
                dataloader_r=retain_dataloader,
                train_dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                mi=mi,
                mine_optimizers=mine_optim,
                groups_info=groups_info,
                epoch=epoch,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )
        elif args.method in ["TR"]:

            loss = train_epoch(
                dataloader=retain_dataloader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                args=args,
                processor=processor,
                time_budget=time_budget,
            )

    save_dir = save_checkpoint(splits, accelerator, args, model, processor)
    return model, save_dir
