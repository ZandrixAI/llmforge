# Copyright © 2024 Apple Inc.


import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .callbacks import TrainingCallback
from .datasets import CacheDataset
from .utils import tree_flatten, tree_map


def _clear_cache(threshold: int):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).forward

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(*args, **kwargs):
            return fn(model, *args, **kwargs)

        return torch.utils.checkpoint.checkpoint(
            inner_fn, *args, use_reentrant=False, **kwargs
        )

    type(layer).forward = checkpointed_fn


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )
    grad_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of steps to accumulate gradients before applying an optimizer update."
        },
    )
    clear_cache_threshold: int = field(
        default=0,
        metadata={
            "help": "Clear the allocator cache between steps if it grows too large."
        },
    )


def default_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = torch.arange(1, targets.shape[1] + 1, device=targets.device)
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])

    ce = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    ce = ce * mask.float()
    ntoks = mask.sum()
    ce = ce.float().sum() / ntoks

    return ce, ntoks


def iterate_batches(
    dataset,
    batch_size,
    max_seq_length,
    loop=False,
    seed=None,
    comm_group=None,
):
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx][0])
    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    if seed:
        np.random.seed(seed)
    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)
            lengths = [len(x) for x in batch]
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = truncated_length
            batch = torch.from_numpy(batch_arr)
            yield batch, torch.tensor(list(zip(offsets, lengths)))

        if not loop:
            break


class _DistributedStub:
    """Stub for distributed that always returns rank=0, size=1."""

    @staticmethod
    def init():
        return _DistributedStub()

    def rank(self):
        return 0

    def size(self):
        return 1


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    clear_cache_threshold: int = 0,
):
    model.eval()
    device = next(model.parameters()).device
    all_losses = torch.tensor(0.0, device=device)
    ntokens = torch.tensor(0, device=device)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    num_batches_eff = (
        min(len(dataset) // batch_size, num_batches)
        if num_batches > 0
        else len(dataset) // batch_size
    )

    for _, batch in tqdm(
        zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ),
        desc="Calculating loss...",
        total=num_batches_eff,
    ):
        batch = (
            batch[0].to(device)
            if isinstance(batch, (tuple, list))
            else batch.to(device)
        )
        with torch.no_grad():
            losses, toks = loss(
                model,
                *(
                    [batch[0] if isinstance(batch, (tuple, list)) else batch]
                    + list(batch[1:])
                ),
            )
        all_losses += losses.detach() * toks.detach()
        ntokens += toks.detach()
        _clear_cache(clear_cache_threshold)

    avg_loss = (all_losses / ntokens).item()

    return avg_loss


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    print(f"Starting training..., iters: {args.iters}")
    device = next(model.parameters()).device
    rank = 0
    world_size = 1

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    grad_accum_steps = args.grad_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0

    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
        ),
    ):
        tic = time.perf_counter()
        if val_dataset and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: Val loss {val_loss:.3f}, Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it - 1,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        batch_device = (
            batch[0].to(device)
            if isinstance(batch, (tuple, list))
            else batch.to(device)
        )
        lvalue, toks = loss(model, *[b.to(device) for b in batch])

        lvalue = lvalue / grad_accum_steps
        lvalue.backward()

        if it % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        losses += lvalue.item() * grad_accum_steps
        n_tokens += toks.item()
        steps += 1
        _clear_cache(args.clear_cache_threshold)
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = losses / steps
            learning_rate = optimizer.param_groups[0]["lr"]
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
            peak_mem = 0
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            _save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            _save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        _save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")


def _save_safetensors(path, weights):
    from safetensors.torch import save_file

    save_file(weights, path)
