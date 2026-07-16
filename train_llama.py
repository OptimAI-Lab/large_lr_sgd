import sys

import os
import time
import random
import argparse

from safetensors.torch import load_file
import json

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error() 

import torch.distributed as dist

from torch.utils.data import IterableDataset

import torch
import torch.utils.data

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import datasets
import datasets.distributed
from datasets import load_dataset

from modeling_llama import LlamaForCausalLM


import wandb
import numpy as np

 
from training_utils.scheduler_utils import get_scheduler



# ---- Ray imports
import ray
from ray import train
from ray.train.torch import TorchTrainer, get_device, prepare_model
from ray.train import RunConfig, ScalingConfig


import training_utils.opt_utils as opt_utils


# Hugging Face logging
transformers.logging.set_verbosity_error()

# (Keep your SDP toggles)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def _get_ray_rank_and_world():
    """Returns (rank, world_size) if under Ray Train; otherwise (0, 1)."""
    try:
        ctx = train.get_context()
        if ctx is None:
            return 0, 1
        return ctx.get_world_rank(), ctx.get_world_size()
    except Exception:
        return 0, 1


# -------------------------
# Dataset utilities
# -------------------------

class PreprocessedIterableDataset(IterableDataset):
    """
    Streaming dataset wrapper that applies tokenizer + batching.
    Sharding is handled outside (via HF `.shard()`).
    """
    def __init__(self, data, tokenizer, batch_size, max_length, start_tokenizing_idx=None):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.start_tokenizing_idx = start_tokenizing_idx
        self.k = 0

    def __iter__(self):
        iter_data = iter(self.data)
        batch = []
        for example in iter_data:
            if self.start_tokenizing_idx is None or self.k > self.start_tokenizing_idx :
                tokenized_example = self.tokenizer(
                    example["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch.append(tokenized_example)
            else:
                batch.append(0)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []
                self.k += 1

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        if self.start_tokenizing_idx is None or self.k > self.start_tokenizing_idx:
            input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
            attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return 0


class TokenizedIterableDataset(IterableDataset):
    """
    Streaming dataset wrapper that applies batching.
    Sharding is handled outside (via HF `.shard()`).
    """
    def __init__(self, data, tokenizer, batch_size, max_length, world_size, rank, start_tokenizing_idx=None):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.world_size = world_size
        self.rank = rank
        
        self.start_tokenizing_idx = start_tokenizing_idx
        self.k = 0
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        iter_data = iter(self.data)
        batch = []
        for i, example in enumerate(iter_data):
            if i % self.world_size != self.rank:
                continue
            if self.start_tokenizing_idx is None or self.k > self.start_tokenizing_idx :
                tokenized_example = example
                batch.append(tokenized_example)
            else:
                batch.append(0)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []
                self.k += 1

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        if self.start_tokenizing_idx is None or self.k > self.start_tokenizing_idx:
            input_ids = torch.tensor([item["input_ids"] + [self.tokenizer.pad_token_id] * (self.max_length - len(item["input_ids"])) for item in batch])
            return {"input_ids": input_ids}
        else:
            return 0

def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def prepare_val_data(args, preprocess_batched):
    if not args.hf_dataset:
        data_files_val= {"validation": [f"{args.dataset_path}/c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0,8)]}
        val_data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_val, split="validation", streaming=True)
    else:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        ) 

    val_data = val_data.shuffle(seed=args.seed) 

    rank, world_size = _get_ray_rank_and_world()
    
    val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=rank, world_size=world_size
        )
    
    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)
    return val_data_mapped


class AverageMeter:
    """Keeps track of the average of a float value."""
    
    def __init__(self):
        self.reset()
        self.dtype = torch.float64
    
    def reset(self):
        """Reset the meter."""
        self.avg = 0.0
        self.count = 0
    
    def update(self, avg, n):
        """
        Update the average with a new average value and its sample count.
        
        Args:
            avg: The new average value to incorporate
            n: The number of samples this average represents
        """
        # Combine old and new averages using weighted sum
        self.avg = (self.avg * self.count + avg.to(self.dtype) * n) / (self.count + n)
        self.count += n
    
    def all_reduce(self):
        rank, world_size = _get_ray_rank_and_world()
        gathered_avg = [torch.zeros_like(self.avg) for _ in range(world_size)]
        dist.all_gather(gathered_avg, self.avg)
        if not isinstance(self.count, torch.Tensor):
            self.count = torch.tensor(self.count, dtype=torch.int64, device=self.avg.device)
        gathered_count = [torch.zeros_like(self.count) for _ in range(world_size)]
        dist.all_gather(gathered_count, self.count)

        for i in range(world_size):
            if i == rank:
                continue
            self.update(gathered_avg[i], gathered_count[i])
        return self.avg


@torch.no_grad()
def evaluate_model(model, val_data_mapped, pad_idx, device, batch_size, args):
    _, world_size = _get_ray_rank_and_world()
    target_eval_tokens = 10_000_000 // world_size
    loss_meter = AverageMeter()

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if loss_meter.count > target_eval_tokens:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        

        if args.amp and not args.eval_in_fp32:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_output = model(**batch, labels=labels)
                loss = model_output.loss
        else:
            model_output = model(**batch, labels=labels)
            loss = model_output.loss        

        n_tokens = (labels[..., 1:] != -100).sum()
        loss_meter.update(loss.detach(), n_tokens)

    loss_meter.all_reduce()


    return loss_meter.avg, loss_meter.count



# -------------------------
# Argparse stays (used on driver)
# -------------------------
def parse_args(args):
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--update_limit", type=float, default=5e-4)
    parser.add_argument("--col_thresh", type=float, default=1e-3)
    
    
    parser.add_argument("--scheduler_load_off", default=False, action="store_true")
    parser.add_argument("--start_tokenizing_idx", type=int, default=None)
    
    parser.add_argument("--keep_only_last_model", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--save_every", type=int, default=-1)

    parser.add_argument("--target_save_step", type=int, default=-1)

    parser.add_argument("--save_logs_every", type=int, default=999999)
    
    parser.add_argument("--continue_from", type=str, default=None)
    
    parser.add_argument("--eval_in_fp32", action='store_true') 

    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler", )
    
    parser.add_argument( "--recovery_steps", type=int, default=10,  help="Number of steps for cosine restarts (only used for cosine_restarts)",)    
    
    parser.add_argument( "--scheduler",  type=str,  default="cosine",  choices=["linear", "cosine", "cosine_restarts","cosine_quick_recovery", "warmup_constant", "wsd_quick_recovery", "exp_cosine"], )
    parser.add_argument( "--decay_type",  type=str,  default="exponential", choices=["exponential", "linear"])
    
    
    parser.add_argument("--amp", action="store_true", help="Enable PyTorch AMP mixed precision training")

    parser.add_argument("--grad_clipping_norm", type=float, default=0.0)
    
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_run_id", type=str)

    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)

    parser.add_argument("--momentum", type=float, default=0.9) 
    parser.add_argument("--beta_1", type=float, default=0.9) 
    parser.add_argument("--beta_2", type=float, default=0.999) 

    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--compile_model", action='store_true')  
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--hf_dataset", default=False, action="store_true")

    parser.add_argument("--dataset_path", type=str, required=True)
    
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0 )
    parser.add_argument("--decay_steps", type=int, default=0 )
    parser.add_argument("--eval_every", type=int )
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int,
                        help="Number of **update steps** to train for.")
    parser.add_argument("--full_bf16", default=False, action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="PyTorch DataLoader workers per Ray worker")
    parser.add_argument("--seed", type=int, default=42)

    # Ray scaling (driver-side only)
    parser.add_argument("--ray_num_workers", type=int, default=2, help="Ray Train workers (processes)")
    parser.add_argument("--ray_use_gpu", action="store_true", help="Set if each Ray worker should use 1 GPU")
    parser.add_argument("--ray_cpus_per_worker", type=int, default=2, help="CPUs per Ray worker for DataLoader")

    return parser.parse_args(args)


# -------------------------
# Ray Train worker loop
# -------------------------
def training_loop_per_worker(config):
    # Pull config into a simple namespace-ish dict
    args = argparse.Namespace(**config)

    # Seeding (per worker)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device from Ray Train (binds correct local_rank / CUDA)
    device = get_device()
    
    rank, world_size = _get_ray_rank_and_world()


    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"
    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"
    

    model_config = AutoConfig.from_pretrained(args.model_config)


    # -------------------------
    # Data: streaming C4
    # -------------------------

    is_rank0 = train.get_context().get_world_rank() == 0
    

    if is_rank0:
        if args.continue_from is not None:
            wandb.init(entity=args.wandb_entity, project=args.wandb_project_name, id=args.wandb_run_id, name=args.model_name)
            wandb.config.update(config, allow_val_change=True)      
        else:
            wandb.init(entity=args.wandb_entity, project=args.wandb_project_name, name=args.model_name)
            wandb.config.update(config, allow_val_change=True)            

    if not args.hf_dataset:
        data_files_train = {"train": [f"{args.dataset_path}/c4-train.{str(i).zfill(5)}-of-01024.json.gz" for i in range(0,1024)]}
        data = load_dataset(path=args.dataset_path,  data_files=data_files_train, split="train", streaming=True)
        data = data.shuffle(seed=args.seed, buffer_size=100_000) 
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True ) 
        data = data.shuffle(seed=args.seed, buffer_size=100_000) 

    if args.continue_from is not None:
        with open(os.path.join(args.continue_from, "training_state.json")) as f:
            _old_state = json.load(f)
            global_step = _old_state["global_step"]
        
        data = data.skip(global_step * args.gradient_accumulation * args.batch_size * world_size)

    data = data.shard(num_shards=world_size, index=rank)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch



    dataset = PreprocessedIterableDataset(
        data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, start_tokenizing_idx = args.start_tokenizing_idx
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers
    )

    val_data_mapped = prepare_val_data(args, preprocess_batched)

    rank = train.get_context().get_world_rank()


    # -------------------------
    # Model
    # -------------------------
    
    
    if args.use_hf_model:
        model = AutoModelForCausalLM.from_config(model_config)
        print(model)

    else:
        model = LlamaForCausalLM(model_config)
        print(model)
    
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()


    save_logs = []
    
    # dtype & device
    if args.full_bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)


    trainable_params = opt_utils.get_trainable_parameters(model, args)
        
    optimizer = opt_utils.build_optimizer(model, trainable_params, args)



    scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,  # Use total steps
            warmup_steps=args.warmup_steps, 
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=args.cycle_length,  # Restart interval
            recovery_steps=args.recovery_steps,
            decay_steps=args.decay_steps,
            decay_type=args.decay_type,
        )   
    


    global_step = 0
    update_step = 0
    tokens_seen = 0
  

    if args.continue_from is not None:

        print("*" * 40)
        print(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        
        if not os.path.exists(checkpoint_path): #safetensors -> bin  
            safetensors_file = os.path.join(args.continue_from, "model.safetensors")
            state_dict = load_file(safetensors_file)
            torch.save(state_dict, checkpoint_path)
 
            print(f"safetensors {safetensors_file} converted to pytorch bin {checkpoint_path}")
        
        
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True,
        )
        print(f"Model successfully loaded (strict=True policy)")

        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu", weights_only=False,
        )
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])


        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            print(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            print(f"global_step       : {global_step}")
            print(f"update_step       : {update_step}")
            print(f"tokens_seen       : {tokens_seen}")
            print(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )
        else:
            print(
                f"Did not find training state in {args.continue_from}, global step will start from zero"
            )
        print("*" * 40)        


        if not args.scheduler_load_off:
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
            print(f"Optimizer and scheduler restored from {args.continue_from}")
  
                    
        else:
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
                param_group["initial_lr"] = args.lr
                print("reset lr = ", param_group["lr"])
                
            
            scheduler = get_scheduler(
                    optimizer=optimizer,
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps,
                    warmup_steps=args.warmup_steps, 
                    min_lr_ratio=args.min_lr_ratio,
                    cycle_length=args.cycle_length,
                    recovery_steps=args.recovery_steps,
                    last_epoch=update_step
                )   
            
            # remaining steps = args.num_training_steps - update_step
    
            
        print(f"Optimizer restored from {args.continue_from}")



    if args.compile_model:
        model = torch.compile(model, mode=args.compile_mode)
        print(f"compiled model, mode = {args.compile_mode}")

    model = prepare_model(model)

    pad_idx = tokenizer.pad_token_id
    local_step = 0

    max_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    
    # Progress bar only on rank 0
    is_rank0 = train.get_context().get_world_rank() == 0

    pbar = None
    if is_rank0:
        from tqdm import tqdm

        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps",
                ncols=80, leave=True, position=0,
                dynamic_ncols=False, ascii=True, file=sys.stdout)


    # -------------------------
    # TRAINING LOOP
    # -------------------------


    train_loss = 0

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum()


        if args.amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):

              
                outputs = model(**batch, labels=labels)

                (outputs.loss / args.gradient_accumulation).backward()

                loss = outputs.loss
                train_loss += outputs.loss / args.gradient_accumulation

        else:
            outputs = model(**batch, labels=labels)
            (outputs.loss / args.gradient_accumulation).backward()
            loss = outputs.loss
            train_loss += outputs.loss / args.gradient_accumulation



        if global_step % args.gradient_accumulation != 0:
            continue
        


        if args.grad_clipping_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping_norm)

        
        optimizer.step()
        scheduler.step()
       
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

        cpu_train_loss = train_loss.cpu()

        train_loss.zero_()

        optimizer.zero_grad()


        update_step += 1


        if  is_rank0 and pbar is not None:
            pbar.update(1)

        lr = optimizer.param_groups[0]["lr"]

        reduced_tokens_seen = tokens_seen.clone()
        dist.all_reduce(reduced_tokens_seen, op=dist.ReduceOp.SUM) # only reduce before logging
        if is_rank0 and wandb.run is not None :
            if (update_step-1) % args.log_every == 0 or update_step == args.num_training_steps:
                save_log = {
                    "iteration": update_step,
                    "lr": lr,
                    "loss": loss.item(),
                    "train_loss": cpu_train_loss.item(),
                    "tokens_seen": reduced_tokens_seen.item(),
                }
                save_logs.append(save_log)

            wandb_logs = {
                "lr": lr,
                "loss": loss.item(),
                "train_loss": cpu_train_loss.item(),
                "update_step": update_step,
                "tokens_seen": reduced_tokens_seen.item(),
                "max_memory": max_memory,
            }
          


            wandb.log(
                wandb_logs,
                step=update_step
            )

        # save checkpoint by save_logs_every
        if (
            local_step > args.gradient_accumulation
            and (update_step % args.save_logs_every == 0 or update_step == args.num_training_steps)
            and is_rank0
            and args.save_dir is not None
        ):
            os.makedirs(args.save_dir, exist_ok=True)
            save_log_target = os.path.join(args.save_dir, "training_logs_{}.pt".format(update_step))
            torch.save(save_logs, save_log_target)
            print("{} saved.".format(save_log_target))
            save_logs = []
 
        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and args.save_every != -1
            and (update_step % args.save_every == 0 or update_step == args.num_training_steps or (args.target_save_step != -1 and update_step == args.target_save_step))
            and is_rank0
            and args.save_dir is not None
        ):
            if args.target_save_step != -1 and update_step == args.target_save_step:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
            elif args.keep_only_last_model:
                current_model_directory = f"{args.save_dir}/model_last"
            else:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
            print(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(
                current_model_directory, max_shard_size="100GB", safe_serialization=False 
            )                

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen.item(),
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
 
 
        # Evaluation
        if args.eval_every > 0 and ((update_step -1) % args.eval_every == 0 or (update_step == args.num_training_steps) ): 
            
            with torch.no_grad():
                total_loss, evaluated_on_tokens = evaluate_model(
                    model, val_data_mapped, pad_idx, device, args.batch_size, args
                )
 
            if is_rank0 and ( wandb.run is not None):
                print(f"[Eval Step {update_step}] Loss: {total_loss:.4f}, PPL: {total_loss.exp():.2f}", flush=True)
                wandb.log(
                                {
                                    "final_eval_loss": total_loss,
                                    "final_eval_perplexity": total_loss.exp(),
                                    "final_eval_tokens": evaluated_on_tokens,
                                },
                                step=update_step
                            ) 
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        if update_step >= args.num_training_steps:
            break
                
 

    # Close progress bar
    if is_rank0 and pbar is not None:
        pbar.close()
 

    # Cleanup
    del loss, optimizer, scheduler
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if is_rank0 and update_step == args.num_training_steps and wandb.run is not None:
        wandb.finish(exit_code=0, quiet=True)
        
        wandb.teardown()
        
        time.sleep(5) 



def main_driver(args):
    # Build Ray Train config dictionary passed to each worker
    worker_config = vars(args)

    # Start Ray locally if not already connected to a cluster
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)  # connects to cluster if RAY_ADDRESS set / head running

    scaling = ScalingConfig(
        num_workers=args.ray_num_workers,
        use_gpu=args.ray_use_gpu,
        resources_per_worker={"CPU": args.ray_cpus_per_worker},  # optional; helps DataLoader perf
    )


    trainer = TorchTrainer(
        training_loop_per_worker,
        train_loop_config=worker_config,
        scaling_config=scaling,
        run_config=RunConfig(name="llama_c4_stream_train"),
    )

    trainer.fit()


if __name__ == "__main__":
    print("Starting Ray script")
    cli_args = parse_args(None)
    main_driver(cli_args)