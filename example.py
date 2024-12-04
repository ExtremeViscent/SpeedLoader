import logging
from argparse import ArgumentParser

import deepspeed
import torch
from deepspeed.comm.utils import get_world_size_from_launcher
from tqdm import trange
from transformers import AutoConfig, LlamaForCausalLM

from speedloader.llama_sl import *
from speedloader.utils import AsyncHiddenSwapper

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--sub-batch-size", type=int, default=2)
    parser.add_argument("--num-sub-batches", type=int, default=8)
    parser.add_argument("--num-live-blocks", type=int, default=1)
    parser.add_argument("--session-name", type=str, default="default")
    parser.add_argument("--vanilla", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model-name-or-path", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B")
    return parser.parse_args()


def report_arguments(args):
    logger = logging.getLogger("ArgumentSettings")
    logger.setLevel(logging.INFO)
    logger.info("Argument settings:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")


def initialize_deepspeed(args, config):
    sub_batch_size = args.sub_batch_size
    num_sub_batches = args.num_sub_batches if not args.vanilla else 1
    num_live_blocks = args.num_live_blocks
    vanilla_batch_size = args.batch_size
    vanilla = args.vanilla
    world_size = get_world_size_from_launcher()
    gradient_accumulation_steps = args.gradient_accumulation_steps
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    layer_numel = config.hidden_size**2 * 4 + config.hidden_size * config.intermediate_size * 3
    if not vanilla:
        batch_size_per_gpu = sub_batch_size * num_sub_batches
        batch_size = sub_batch_size * num_sub_batches * world_size * gradient_accumulation_steps
        num_ckpts = config.num_hidden_layers // num_live_blocks + 3
        num_ckpts += 1 if config.num_hidden_layers % num_live_blocks != 0 else 0
    else:
        batch_size = vanilla_batch_size * world_size * gradient_accumulation_steps
        batch_size_per_gpu = vanilla_batch_size
    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "betas": [0.9, 0.98],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "fast_init": True,
                "nvme_path": "/nvme",
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
                "nvme_path": "/nvme",
                "buffer_size": int(config.hidden_size * config.vocab_size * 2.5),
                "max_in_cpu": 1e9,
                "buffer_count": 8,
            },
            "stage3_max_live_parameters": int(num_live_blocks * layer_numel * 2),
            "stage3_max_reuse_distance": int(num_live_blocks * layer_numel),
            "stage3_prefetch_bucket_size": int(num_live_blocks * layer_numel),
            "reduce_bucket_size": int(num_live_blocks * layer_numel),
            "zero_hpz_partition_size": 1,
        },
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "zero_allow_untested_optimizer": True,
    }
    return ds_config, batch_size_per_gpu


def initialize_model(args, config, ds_config):
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = (
            LlamaForCausalLMSL(
                config=config, batch_size=args.sub_batch_size * args.num_sub_batches, sub_batch_size=args.sub_batch_size
            )
            if not args.vanilla
            else LlamaForCausalLM(config)
        )
        model.gradient_checkpointing = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.use_cache = False
        model.config.use_cache = False
    if not args.vanilla:
        model.model.output_logits = False
        model.model.n_live_blocks = args.num_live_blocks
        model.model.sub_batch_size = args.sub_batch_size
        model.model.num_ckpts = model.model.n_layers // args.num_live_blocks + 3
        model.model.num_ckpts += 1 if model.model.n_layers % model.model.n_live_blocks != 0 else 0
    return model


def initialize_engine(model, ds_config):
    return deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )


def setup_speedloader(args, config, seq_len, num_sub_batches, sub_batch_size, num_ckpts, model):
    swapper = (
        AsyncHiddenSwapper(
            config.hidden_size,
            seq_len,
            num_sub_batches * sub_batch_size,
            sub_batch_size,
            num_ckpts,
            True,
            dtype=torch.bfloat16,
            quantization=False,
        )
        if not args.vanilla
        else None
    )
    model.model.hidden_swapper = swapper


def train_loop(engine, args, config):
    batch_size_per_gpu = args.batch_size_per_gpu
    seq_len = args.sequence_length
    logger = logging.getLogger("Training")
    for i in trange(3):
        engine.train()
        engine.zero_grad()
        for j in range(args.gradient_accumulation_steps):
            if i == 0 and j == 0:
                logger.info("Warmup")
            batch = {
                "input_ids": torch.randint(
                    0, config.vocab_size, (batch_size_per_gpu, seq_len), dtype=torch.long, device=engine.local_rank
                ),
                "attention_mask": torch.ones((batch_size_per_gpu, seq_len), dtype=torch.long, device=engine.local_rank),
                "labels": torch.randint(
                    0, config.vocab_size, (batch_size_per_gpu, seq_len), dtype=torch.long, device=engine.local_rank
                ),
            }
            engine.zero_grad()
            outputs = engine(**batch)
            loss = outputs[0]
            engine.backward(loss)
            if i == 0 and j == 0:
                logger.info("Warmup complete")


def main():
    deepspeed.init_distributed()
    args = parse_arguments()
    report_arguments(args)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    ds_config, batch_size_per_gpu = initialize_deepspeed(args, config)
    args.batch_size_per_gpu = batch_size_per_gpu
    model = initialize_model(args, config, ds_config)
    engine, optimizer, _, _ = initialize_engine(model, ds_config)
    setup_speedloader(
        args, config, args.sequence_length, args.num_sub_batches, args.sub_batch_size, model.model.num_ckpts, model
    )
    train_loop(engine, args, config)


if __name__ == "__main__":
    main()
