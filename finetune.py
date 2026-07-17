import copy

import torch
from torch.utils.data import DataLoader


from data_process.data_classes import (
    IDE_train_Dataset,
    VQA_train_Dataset,
    train_collate_fn_qwen_mixed,
)
from methods import finetune_epoch, train_epoch
from utils import save_checkpoint


def finetune(
    args,
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    splits,
    target_size,
    collate_fn,
    processor,
    val_dataset=None,
):

    multimodal_dataset = IDE_train_Dataset(
        args.hf_dataset,
        target_size=target_size,
        train_ids=splits["retain"],
        mode="train",
        log=accelerator.is_main_process,
    )  # TODO: REMOVE TS

    if args.mixed > 0:
        nsamples = int((len(multimodal_dataset) * args.mixed))
        vqav2_dataset = VQA_train_Dataset(nsamples=nsamples)
        multimodal_dataset = torch.utils.data.ConcatDataset(
            [multimodal_dataset, vqav2_dataset]
        )
        collate_fn = train_collate_fn_qwen_mixed

    train_dataloader = DataLoader(
        multimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=4 * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    for epoch in range(args.num_epochs):
        model.train()

        loss = finetune_epoch(
            dataloader_t=train_dataloader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            args=args,
            processor=processor,
            val_dataset=val_dataset,
        )

        if epoch < args.num_epochs - 1 and args.checkpoint:
            save_checkpoint(
                splits, accelerator, args, model, processor, epoch=epoch + 1
            )
    save_dir = save_checkpoint(splits, accelerator, args, model, processor)

    return model, save_dir
