"""
Training script for Titans Language Model.
Optimized for Multi-GPU (DDP) on H100/H200 clusters.

Features:
- Distributed Data Parallel (DDP)
- Flash Attention 2 + BFloat16
- torch.compile support
- Streaming dataset sharding
"""

import os
import math
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
from typing import Optional, Dict, Iterator, Any
from itertools import islice
import argparse
from tqdm import tqdm
import time

# Use TF32 on Ampere/Hopper
torch.set_float32_matmul_precision('high')

from titans import TitansLM, TitansConfig

def setup_distributed():
    """Initialize DDP."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_tokenizer(name: str = "gpt2"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_dataset_no_remote_code(*args, dataset_name_for_error: str, **kwargs):
    from datasets import load_dataset
    try:
        return load_dataset(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "trust_remote_code" in msg or "remote code" in msg:
            print(f"Warning: {dataset_name_for_error} requires remote code. trying to proceed...")
            raise e
        raise

class StreamingTextDataset(IterableDataset):
    """DDP-aware streaming dataset."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        tokenizer,
        seq_len: int = 1024,
        split: str = "train",
        text_column: str = "text",
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.text_column = text_column
        self.rank = rank
        self.world_size = world_size
        self.max_samples = max_samples
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        load_kwargs = {"split": self.split, "streaming": True}
        if self.dataset_config:
            load_kwargs["name"] = self.dataset_config
            
        dataset = load_dataset_no_remote_code(
            self.dataset_name, **load_kwargs, dataset_name_for_error=self.dataset_name
        )
        
        # 1. Shard by process (DDP)
        # 2. Shard by worker (DataLoader workers)
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker
            iter_start = self.rank
            iter_step = self.world_size
        else:
            # Multiple workers: global stride = world_size * num_workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_start = self.rank * num_workers + worker_id
            iter_step = self.world_size * num_workers
            
        iterator = islice(dataset, iter_start, None, iter_step)
        
        buffer = []
        sample_count = 0
        for example in iterator:
            # Check if we've reached the sample limit
            if self.max_samples is not None and sample_count >= self.max_samples:
                break
                
            text = example.get(self.text_column, "")
            if not text:
                continue
                
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                # Check limit again before yielding
                if self.max_samples is not None and sample_count >= self.max_samples:
                    break
                    
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                sample_count += 1
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long)
                }

class TitansTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        lr: float,
        max_steps: int,
        grad_accum_steps: int,
        save_dir: str,
        log_every: int,
        rank: int,
    ):
        self.model = model
        self.train_loader = train_loader
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        try:
            import bitsandbytes as bnb
            print("Using 8-bit AdamW optimizer")
            self.optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
        except ImportError:
            print("bitsandbytes not found, using standard AdamW")
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True
            )
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, max_steps)
        
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.rank = rank
        self.save_dir = save_dir
        self.log_every = log_every
        
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            
    def train(self):
        self.model.train()
        # Enable gradient checkpointing on the backbone specifically
        if hasattr(self.model, "module"):
            backbone = self.model.module.backbone
        else:
            backbone = self.model.backbone
            
        # Enable gradient checkpointing if available
        if hasattr(backbone, "gradient_checkpointing_enable"):
             backbone.gradient_checkpointing_enable()
             
        step = 0
        accum_loss = torch.tensor(0.0, device="cuda")
        
        data_iter = iter(self.train_loader)
        if self.rank == 0:
            pbar = tqdm(total=self.max_steps, desc="Training")
        
        start_time = time.time()
        
        while step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            
            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                
                # BFloat16 context
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"] / self.grad_accum_steps
                
                loss.backward()
                accum_loss += loss.detach()
            
            # Clip grad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            step += 1
            
            # Logging
            if step % self.log_every == 0:
                # Reduce loss across GPUs for logging
                if dist.is_initialized():
                    dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)
                
                if self.rank == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    tps = (self.log_every * self.grad_accum_steps * 
                           batch["input_ids"].shape[0] * batch["input_ids"].shape[1] * 
                           (dist.get_world_size() if dist.is_initialized() else 1)) / (time.time() - start_time)
                    
                    pbar.set_postfix({"loss": f"{accum_loss.item():.4f}", "tok/s": f"{int(tps):,}"})
                    pbar.update(self.log_every)
                    start_time = time.time()
                
                accum_loss.fill_(0.0)
            
            if step % 1000 == 0 and self.rank == 0:
                self.save_checkpoint(step)

        if self.rank == 0:
            self.save_checkpoint(step, final=True)
            pbar.close()

    def save_checkpoint(self, step, final=False):
        # Unwrap DDP
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # Unwrap Compile
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
            
        path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        torch.save(model_to_save.state_dict(), path)
        print(f"\nSaved checkpoint {path}")

def estimate_vram_gb(d_model: int, n_layers: int, seq_len: int, batch_size: int, quantize: str) -> float:
    """Estimate VRAM usage in GB for given config (Titans-specific).
    
    CRITICAL: Titans memory module is extremely memory-hungry because:
    1. Each layer stores per-batch weight matrices (B, Out, In) 
    2. Memory update computes gradients for each timestep in the chunk
    3. Gradient computation creates intermediate tensors
    """
    hidden_mult = 2.0  # Memory MLP hidden multiplier
    memory_depth = 2   # Memory MLP layers
    chunk_size = 8     # Default chunk size for memory updates
    
    hidden_dim = int(d_model * hidden_mult)
    
    # === Model Parameters ===
    # Standard transformer params per layer
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * hidden_dim  # up + down
    
    # Memory module params per layer
    memory_mlp_params = d_model * hidden_dim + hidden_dim * d_model  # 2-layer MLP
    memory_proj_params = 3 * d_model * d_model  # W_K, W_V, W_Q
    
    params_per_layer = attn_params + ffn_params + memory_proj_params + memory_mlp_params
    total_params = n_layers * params_per_layer + d_model * 50257
    
    bytes_per_param = 0.5 if quantize != "none" else 2
    model_gb = (total_params * bytes_per_param) / 1e9
    
    # === Optimizer States (8-bit AdamW) ===
    optimizer_gb = (total_params * 2) / 1e9
    
    # === CRITICAL: Titans Memory State per Layer ===
    # Each layer stores: weights (B, Out, In) + momentum (B, Out, In) for each MLP layer
    # Layer 0: (B, hidden_dim, d_model) 
    # Layer 1: (B, d_model, hidden_dim)
    layer0_size = batch_size * hidden_dim * d_model
    layer1_size = batch_size * d_model * hidden_dim
    memory_weights_per_layer = (layer0_size + layer1_size) * 2  # bf16 bytes
    memory_momentum_per_layer = memory_weights_per_layer  # same size
    
    memory_state_gb = (n_layers * (memory_weights_per_layer + memory_momentum_per_layer)) / 1e9
    
    # === CRITICAL: Memory Update Gradients ===
    # During update_memory_chunk, for EACH timestep in chunk:
    # - Compute forward through MLP
    # - Compute gradients for all weight tensors  
    # - These gradients are same size as weights
    # PyTorch keeps intermediate tensors for backward pass
    n_chunks = seq_len // chunk_size
    
    # Gradient tensors per update (same size as weights)
    grad_per_update = memory_weights_per_layer
    
    # Intermediate activations during gradient computation
    # For each chunk: forward activations + backward intermediates
    chunk_activations = batch_size * chunk_size * d_model * 4  # Multiple intermediate tensors
    
    # Total memory update overhead (this is the killer!)
    memory_update_gb = (n_layers * (grad_per_update * 2 + chunk_activations * 2)) / 1e9
    
    # === Standard Activations ===
    # Attention: Q, K, V, scores, output per layer
    activation_gb = (batch_size * seq_len * d_model * n_layers * 8 * 2) / 1e9
    
    # === Gradients for model params ===
    grad_gb = model_gb
    
    # === Overhead ===
    overhead_gb = 6.0  # CUDA allocator fragmentation, etc
    
    total = model_gb + optimizer_gb + memory_state_gb + memory_update_gb + activation_gb + grad_gb + overhead_gb
    return total

def auto_fit_vram(vram_gb: float, d_model: int, n_layers: int, seq_len: int, quantize: str):
    """Auto-calculate batch_size and grad_accum to fit in VRAM.
    
    Uses empirically-tested configurations for Titans architecture.
    The memory module is extremely VRAM-hungry due to per-batch weight matrices.
    """
    # Target 75% utilization (Titans needs headroom for memory state updates)
    target_vram = vram_gb * 0.75
    
    # Empirically tested configs for d_model=1024, n_layers=16
    # Format: (batch_size, seq_len, approx_vram_gb)
    # UPDATED for FastNeuralMemory (much lower memory usage!)
    if quantize != "none":
        # 4-bit quantization
        tested_configs = [
            (8, 2048, 45),
            (8, 1024, 30),
            (4, 2048, 28),
            (4, 1024, 18),
            (2, 2048, 16),
            (2, 1024, 10),
        ]
    else:
        # Full precision (bf16) - fast memory uses ~5x less VRAM
        tested_configs = [
            (8, 2048, 55),
            (8, 1024, 35),
            (4, 2048, 32),
            (4, 1024, 20),
            (2, 2048, 18),
            (2, 1024, 12),
            (1, 2048, 10),
        ]
    
    # Scale by model size (relative to reference d_model=1024, n_layers=16)
    size_scale = (d_model / 1024) ** 2 * (n_layers / 16)
    
    for batch_size, test_seq, base_vram in tested_configs:
        scaled_vram = base_vram * size_scale
        if scaled_vram <= target_vram:
            actual_seq = min(seq_len, test_seq)
            # Calculate grad_accum to reach effective batch of 32
            grad_accum = max(1, 32 // batch_size)
            return batch_size, grad_accum, scaled_vram, actual_seq
    
    # Fallback: minimum config
    return 1, 32, 15 * size_scale, 128

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="mag")
    parser.add_argument("--dataset", default="fineweb")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Sequence length (auto-calculated if not set)")
    parser.add_argument("--batch_size", type=int, default=None, 
                        help="Micro batch size (auto-calculated if not set)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (auto-calculated if not set)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--quantize", type=str, default="none", 
                        choices=["none", "nf4", "fp4"],
                        help="4-bit quantization: nf4, fp4, or none")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset to this many samples (for testing or memory constraints)")
    parser.add_argument("--vram", type=int, default=80,
                        help="Available VRAM in GB (default: 80 for H100)")
    args = parser.parse_args()
    
    # Auto-fit to VRAM if key params not specified
    if args.batch_size is None or args.grad_accum is None or args.seq_len is None:
        requested_seq = args.seq_len or 2048
        auto_bs, auto_ga, estimated_vram, auto_seq = auto_fit_vram(
            args.vram, args.d_model, args.n_layers, requested_seq, args.quantize
        )
        if args.batch_size is None:
            args.batch_size = auto_bs
        if args.grad_accum is None:
            args.grad_accum = auto_ga
        if args.seq_len is None:
            args.seq_len = auto_seq
        print(f"🎯 Auto-fit for {args.vram}GB VRAM:")
        print(f"   batch_size={args.batch_size}, grad_accum={args.grad_accum}, seq_len={args.seq_len}")
        print(f"📊 Estimated VRAM usage: {estimated_vram:.1f}GB / {args.vram}GB")
    else:
        # Still show estimate
        estimated = estimate_vram_gb(args.d_model, args.n_layers, args.seq_len, args.batch_size, args.quantize)
        print(f"📊 Estimated VRAM usage: {estimated:.1f}GB / {args.vram}GB")

    # Dist setup
    rank, local_rank, world_size = setup_distributed()
    if rank == 0:
        print(f"🚀 Training on {world_size} GPUs (Rank 0)")

    # Data
    tokenizer = get_tokenizer()
    ds_config = "sample-10BT" if args.dataset == "fineweb" else None
    dataset_name = "HuggingFaceFW/fineweb-edu" if args.dataset == "fineweb" else args.dataset
    
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        dataset_config=ds_config,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        rank=rank,
        world_size=world_size,
        max_samples=args.max_samples
    )
    
    if rank == 0 and args.max_samples:
        print(f"📊 Limiting dataset to {args.max_samples:,} samples")
    
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True
    )
    
    # Model config
    config = TitansConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.d_model // 64,
        variant=args.variant,
        max_seq_len=args.seq_len
    )
    
    # Create model (quantized or full precision)
    if args.quantize != "none":
        if rank == 0:
            print(f"🔢 Using {args.quantize.upper()} 4-bit quantization")
        from titans.quantization import create_quantized_model
        model = create_quantized_model(config, args.quantize)
    else:
        model = TitansLM(config).cuda()
    
    # Compile (optional)
    if args.compile:
        if rank == 0: print("Compiling model...")
        model = torch.compile(model)
    else:
        if rank == 0: print("Skipping torch.compile (use --compile to enable)")
    
    # DDP Wrapper
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
        
    trainer = TitansTrainer(
        model, train_loader, args.lr, args.max_steps, 
        args.grad_accum, "checkpoints", 50, rank
    )
    
    trainer.train()
    cleanup_distributed()

if __name__ == "__main__":
    main()

