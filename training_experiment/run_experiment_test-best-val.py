# run_experiment_with_test_set_and_best_val_iter.py
import os
import time
import glob  # Added for checking existing run files

# --- Imports ---
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets import load_dataset

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.8, rc={"figure.figsize": (8, 6)}, font="monospace",
              context="notebook", color_codes=True)
from datetime import datetime

# --- Logging Setup ---
log_file_path = "experiment_log_with_test_best_val.txt"  # Updated log file name
try:
    log_file = open(log_file_path, "a")
except Exception as e:
    print(f"Error opening log file {log_file_path}: {e}")
    log_file = None  # Fallback if file cannot be opened


def custom_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    message = " ".join(str(a) for a in args)
    full_message = f"{timestamp} {message}"
    print(full_message, **kwargs)
    if log_file:
        try:
            print(full_message, file=log_file)
            log_file.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}")


# --- Entmax Availability Check ---
try:
    from entmax import entmax_bisect

    custom_print("Successfully imported entmax_bisect.")
    ENTMAX_AVAILABLE = True
except ImportError:
    custom_print("Warning: entmax library not found or import failed.")
    custom_print("         AlphaEntmaxRouter will default to Softmax for alpha != 1.0.")
    custom_print("         Install entmax: pip install entmax")
    ENTMAX_AVAILABLE = False


    def entmax_bisect(*args, **kwargs):  # Fallback definition
        custom_print("Fallback: Using Softmax instead of entmax_bisect (for alpha != 1.0).")
        logits = args[0]
        return F.softmax(logits, dim=-1)

# --- Base Configuration () ---
config_base = {
    "learning_rate": 1e-3,
    "dropout": 0.1,
    "eval_iters_scale_factor": 10,  # Factor to scale eval_iters based on max_iters/eval_interval
    "router_alpha": 1.5,
    "router_temperature": 1.0,
    "entmax_n_iter": 25,
    "seed": 42,
    "dataset": "tinyshakespeare",  # Change to "wikitext" for WikiText dataset
}

# --- Sweep Configuration () ---
config_sweep = {
    **config_base,
    "batch_size": 64,
    "block_size": 64,
    "max_iters": 500,  # Reduced for quicker testing, was 5000
    "eval_interval": 20,  # Reduced for quicker testing, was 100
    "n_embed": 64,
    "n_head": 4,
    "n_layer": 3,
    "num_experts": 4,
}
# --- Full Run Configuration (Example, adjust as needed) ---
config_full = {
    **config_base,
    "batch_size": 128,  # Example full run, was 64 for sweep
    "block_size": 128,  # Example full run, was 64 for sweep
    "max_iters": 5000,  # Full run iterations
    "eval_interval": 100,  # Full run eval interval
    "n_embed": 128,  # Example full run
    "n_head": 8,  # Example full run
    "n_layer": 6,  # Example full run
    "num_experts": 8,  # Example full run
    "router_alpha": config_base["router_alpha"],  # Default, will be overridden in sweep
    "router_temperature": config_base["router_temperature"],  # Default, will be overridden in sweep
}

# --- Determine config to use (e.g., sweep or full) ---
# For this script, let's assume we are using config_sweep for demonstration
current_config = config_sweep  # Or config_full if you want to run that
current_config["eval_iters"] = (current_config["max_iters"] // current_config["eval_interval"]) * current_config[
    "eval_iters_scale_factor"]
custom_print(f"Using configuration: {current_config}")
# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else (
    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
custom_print(f"Using device: {device}")

# --- Data Loading and Preprocessing (Tiny Shakespeare) ---
data_file = 'input.txt'

if not os.path.exists(data_file):
    custom_print(f"Error: Dataset file '{data_file}' not found!")
    custom_print(f"Please download Tiny Shakespeare input.txt and place it here.")
    custom_print(
        f"E.g., from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    exit()


def load_shakespeare_dataset():
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        custom_print(f"Successfully loaded {data_file}")

        chars = sorted(list(set(text)))
        full_vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l if i in itos])

        data = torch.tensor(encode(text), dtype=torch.long)
        custom_print(f"Dataset Stats: Vocab size: {full_vocab_size}, Total tokens: {len(data)}")

        # --- MODIFIED DATA SPLIT: 80% train, 10% val, 10% test ---
        n = len(data)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        # n_test is the remainder

        train_data = data[:n_train]
        val_data = data[n_train: n_train + n_val]
        test_data = data[n_train + n_val:]
        # --- END MODIFIED DATA SPLIT ---

        custom_print(
            f"Data Split: Train tokens: {len(train_data)}, Valid tokens: {len(val_data)}, Test tokens: {len(test_data)}")
        return train_data, val_data, test_data, full_vocab_size, stoi, itos
    except Exception as e:
        custom_print(f"Error processing dataset file: {e}")
        exit()


def load_wiki_dataset():
    # --- Data Loading and Preprocessing (Salesforce/wikitext) ---
    try:
        # Load the dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

        # Extract text from each split
        train_text = [item['text'] for item in ds['train']]
        val_text = [item['text'] for item in ds['validation']]
        test_text = [item['text'] for item in ds['test']]

        # Combine text into single strings
        train_text = "\n".join(train_text)
        val_text = "\n".join(val_text)
        test_text = "\n".join(test_text)

        # Build vocabulary and encoding/decoding functions
        full_text = train_text + val_text + test_text
        chars = sorted(list(set(full_text)))
        full_vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l if i in itos])

        # Encode text into tensors
        train_data = torch.tensor(encode(train_text), dtype=torch.long)
        val_data = torch.tensor(encode(val_text), dtype=torch.long)
        test_data = torch.tensor(encode(test_text), dtype=torch.long)

        custom_print(f"Dataset Stats: Vocab size: {full_vocab_size}, Train tokens: {len(train_data)}, "
                     f"Valid tokens: {len(val_data)}, Test tokens: {len(test_data)}")
        return train_data, val_data, test_data, full_vocab_size, stoi, itos
    except Exception as e:
        custom_print(f"Error processing dataset: {e}")
        exit()


if current_config["dataset"] == "tinyshakespeare":
    train_data, val_data, test_data, vocab_size, stoi, itos = load_shakespeare_dataset()
elif current_config["dataset"] == "wikitext":
    train_data, val_data, test_data, vocab_size, stoi, itos = load_wiki_dataset()


# --- Model Definitions (Assuming these are the same as your original run_experiment.py) ---
class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention module """

    def __init__(self, n_embed, num_heads, block_size, dropout):
        super().__init__()
        assert n_embed % num_heads == 0
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Expert(nn.Module):
    """ Simple FeedForward Expert network """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AlphaEntmaxRouter(nn.Module):
    """ Router using alpha-entmax, with explicit Softmax for alpha=1.0 """

    def __init__(self, n_embed, num_experts, alpha=1.5, temperature=1.0, n_iter=25):
        super().__init__()
        assert temperature > 1e-9, "Temperature must be positive"
        self.num_experts = num_experts
        self.alpha = alpha
        self.temperature = temperature
        self.n_iter = n_iter
        self.route_linear = nn.Linear(n_embed, num_experts)
        if not ENTMAX_AVAILABLE and self.alpha != 1.0:
            custom_print(
                f"Warning: AlphaEntmaxRouter initialized with alpha={self.alpha} but entmax library not found. Will use Softmax fallback.")

    def forward(self, x):  # x is the input tensor
        logits = self.route_linear(x)
        scaled_logits = logits / self.temperature
        router_output = None

        if self.alpha == 1.0:
            router_output = F.softmax(scaled_logits, dim=-1)
        elif ENTMAX_AVAILABLE:
            try:
                router_output = entmax_bisect(scaled_logits, alpha=self.alpha, dim=-1, n_iter=self.n_iter)
            except Exception as e:
                custom_print(
                    f"WARNING: entmax_bisect failed with alpha={self.alpha}. Error: {e}. Falling back to Softmax.")
                router_output = F.softmax(scaled_logits, dim=-1)
        else:  # Fallback for alpha != 1.0 if entmax not available
            router_output = F.softmax(scaled_logits, dim=-1)

        if router_output is None:  # Should not happen with the logic above, but as a safeguard
            custom_print("ERROR: router_output logic failed unexpectedly. Defaulting to Softmax.")
            router_output = F.softmax(scaled_logits, dim=-1)

        return router_output


class SparseMoE(nn.Module):
    """ Sparse Mixture of Experts layer """

    def __init__(self, n_embed, num_experts, dropout, router_alpha=1.5, router_temperature=1.0):
        super().__init__()
        self.router = AlphaEntmaxRouter(n_embed, num_experts, alpha=router_alpha, temperature=router_temperature)
        self.experts = nn.ModuleList([Expert(n_embed, dropout) for _ in range(num_experts)])
        self.num_experts = num_experts
        # Buffers for metrics
        self.register_buffer('latest_tpe', torch.zeros(num_experts, dtype=torch.float32))
        self.latest_imbalance: float = 1.0  # Default for safety
        self.latest_concentration: float = 0.0
        self.latest_utilization: float = 0.0
        self.latest_tpe_cv: float = 0.0
        self.latest_avg_prob_cv: float = 0.0
        self.activation_threshold = 1e-9  # Threshold for considering an expert activated

    def calculate_metrics(self, gating_output_no_grad):
        """ Calculates routing metrics based on detached gating output """
        # gating_output_no_grad is expected to be [num_tokens, num_experts]
        num_tokens = gating_output_no_grad.shape[0]
        if num_tokens == 0 or self.num_experts == 0:
            self.latest_tpe.zero_()
            self.latest_imbalance = 1.0
            self.latest_concentration = 0.0
            self.latest_utilization = 0.0
            self.latest_tpe_cv = 0.0
            self.latest_avg_prob_cv = 0.0
            return

        with torch.no_grad():
            # Tokens Per Expert (TPE)
            is_active = gating_output_no_grad > self.activation_threshold  # [num_tokens, num_experts]
            tpe = is_active.sum(dim=0).float()  # Sum active tokens for each expert -> [num_experts]

            # Ensure buffer has the correct shape before copying
            if self.latest_tpe.shape[0] == tpe.shape[0]:
                self.latest_tpe.copy_(tpe)
            else:
                # This case should be rare, but handles potential dynamic changes if num_experts could change
                custom_print(
                    f"Warning: TPE size mismatch during metrics. Re-registering buffer. Expected: {self.latest_tpe.shape[0]}, Got: {tpe.shape[0]}")
                self.register_buffer('latest_tpe', tpe.clone())

            # TPE CV (Coefficient of Variation)
            if self.num_experts > 1:
                mean_tpe = tpe.mean()
                std_tpe = tpe.std(unbiased=False)  # Population standard deviation
                self.latest_tpe_cv = (std_tpe / (mean_tpe + 1e-9)).item() if mean_tpe > 1e-9 else 0.0
            else:
                self.latest_tpe_cv = 0.0  # CV is 0 for a single expert

            # Imbalance
            mean_tpe_val = tpe.mean().item()  # Use .item() to get Python float
            if mean_tpe_val > 0:  # Avoid division by zero
                self.latest_imbalance = tpe.max().item() / (mean_tpe_val + 1e-9)
            else:
                self.latest_imbalance = 1.0  # Perfect balance if no tokens or all experts have zero

            # Concentration (average max probability per token)
            max_p_per_token, _ = gating_output_no_grad.max(dim=-1)  # Max probability for each token -> [num_tokens]
            self.latest_concentration = max_p_per_token.mean().item()

            # Utilization
            num_active_experts = (tpe > 0).sum().item()  # Number of experts that received at least one token
            self.latest_utilization = num_active_experts / self.num_experts if self.num_experts > 0 else 0.0

            # Average Probability CV
            if self.num_experts > 1:
                avg_prob = gating_output_no_grad.mean(
                    dim=0)  # Average probability assigned to each expert -> [num_experts]
                mean_avg_prob = avg_prob.mean()
                std_avg_prob = avg_prob.std(unbiased=False)
                self.latest_avg_prob_cv = (
                        std_avg_prob / (mean_avg_prob + 1e-9)).item() if mean_avg_prob > 1e-9 else 0.0
            else:
                self.latest_avg_prob_cv = 0.0

    def forward(self, x):  # x shape: (B, T, C_embed)
        batch_size, seq_len, n_embed = x.shape
        num_tokens = batch_size * seq_len  # Total tokens in the batch

        x_reshaped = x.view(num_tokens, n_embed)  # Reshape for router: (B*T, C_embed)

        gating_output = self.router(x_reshaped)  # Get routing probabilities: (B*T, num_experts)

        # Calculate and store metrics (using detached tensor for no grad impact)
        self.calculate_metrics(gating_output.detach())

        final_output = torch.zeros_like(x_reshaped)  # Initialize output tensor

        # Iterate over experts
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert (above threshold)
            token_indices = torch.nonzero(gating_output[:, i] > self.activation_threshold).squeeze(-1)

            if token_indices.numel() == 0:  # If no tokens for this expert, skip
                continue

            expert_input = x_reshaped[token_indices]  # Select input tokens for this expert
            active_gating_scores = gating_output[token_indices, i].unsqueeze(1)  # Get scores for these tokens

            expert_output = expert(expert_input)  # Pass tokens through the expert
            weighted_output = expert_output * active_gating_scores  # Weight expert output by routing score

            final_output.index_add_(0, token_indices, weighted_output)  # Add to final output

        final_output = final_output.view(batch_size, seq_len, n_embed)  # Reshape back to original
        return final_output


class Block(nn.Module):
    """ Transformer block: communication followed by computation (MoE) """

    def __init__(self, n_embed, n_head, num_experts, block_size, dropout, router_alpha=1.5, router_temperature=1.0):
        super().__init__()
        assert n_embed % n_head == 0
        self.self_attn = MultiHeadAttention(n_embed, n_head, block_size, dropout)
        self.sparse_moe = SparseMoE(n_embed, num_experts, dropout, router_alpha=router_alpha,
                                    router_temperature=router_temperature)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))  # Self-attention with pre-LayerNorm
        x = x + self.sparse_moe(self.ln2(x))  # SparseMoE with pre-LayerNorm
        return x


class SparseMoELanguageModel(nn.Module):
    """ Language Model using SparseMoE Blocks """

    def __init__(self, vocab_size, n_embed, n_head, n_layer, num_experts, block_size, dropout, router_alpha=1.5,
                 router_temperature=1.0):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)  # Each position gets an embedding
        self.blocks = nn.Sequential(*[
            Block(n_embed=n_embed, n_head=n_head, num_experts=num_experts, block_size=block_size, dropout=dropout,
                  router_alpha=router_alpha, router_temperature=router_temperature)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)  # Final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)  # Linear layer to map to vocab size

    def forward(self, idx, targets=None):  # idx is (B, T) tensor of token indices
        B, T = idx.shape
        device = idx.device  # Get device from input tensor

        tok_emb = self.token_embedding_table(idx)  # (B, T, C_embed)

        # Position embeddings
        pos = torch.arange(T, device=device)  # (T)
        # Handle cases where T might be > block_size during generation/inference if not chunked
        if T > self.block_size:
            custom_print(
                f"Warning: Input sequence length T={T} exceeds block_size={self.block_size}. Truncating pos_emb and input for this forward pass.")
            pos = pos[-self.block_size:]  # Use last block_size positions
            tok_emb = tok_emb[:, -self.block_size:,
                      :]  # Ensure token embeddings are also truncated if input T > block_size
            T = self.block_size  # Update T to reflect truncation for correct logit/target alignment

        pos_emb = self.position_embedding_table(pos)  # (T, C_embed)

        x = tok_emb + pos_emb  # (B, T, C_embed) Add token and position embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Final layer norm
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape logits and targets for cross_entropy
            # Logits: (B, T, C_vocab) -> (B*T, C_vocab)
            # Targets: (B, T) -> (B*T)
            B_logits, T_logits, C_logits = logits.shape  # Get dimensions from logits

            # Ensure targets are shaped correctly relative to logits T_logits
            if targets.shape[1] > T_logits:
                targets_for_loss = targets[:, :T_logits]  # Use the portion of targets that aligns with T_logits
            elif targets.shape[1] < T_logits:
                # This case should ideally not happen if input idx and targets are prepared consistently
                custom_print(
                    f"Warning: Targets length {targets.shape[1]} is less than logits length {T_logits}. This might be an issue.")
                # Decide on handling: either error out or truncate logits (less ideal for loss calc)
                # For now, we'll proceed with available targets, cross_entropy will handle if dimensions mismatch
                targets_for_loss = targets
            else:
                targets_for_loss = targets

            logits_flat = logits.reshape(B_logits * T_logits, C_logits)
            targets_flat = targets_for_loss.reshape(
                B_logits * T_logits)  # Ensure targets are also flattened consistently
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# --- Utility Functions ---
# --- MODIFIED get_batch to handle train, val, and test ---
def get_batch(split, data_train, data_val, data_test, block_size, batch_size, device):
    """ Generate a small batch of data of inputs x and targets y """
    if split == 'train':
        data_source = data_train
    elif split == 'val':
        data_source = data_val
    elif split == 'test':
        data_source = data_test
    else:
        raise ValueError(f"Invalid split name: {split}. Must be 'train', 'val', or 'test'.")

    max_start_index = len(data_source) - block_size - 1  # Ensure there's room for x and y
    if max_start_index < 0:
        custom_print(
            f"Warning: Dataset split '{split}' length ({len(data_source)}) is smaller than block_size+1 ({block_size + 1}). Cannot generate batch.")
        return None, None  # Return None if batch cannot be formed

    ix = torch.randint(max_start_index + 1, (batch_size,))  # Random start indices
    x = torch.stack([data_source[i: i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1: i + block_size + 1] for i in ix])  # Targets are shifted by one
    x, y = x.to(device), y.to(device)
    return x, y


# --- END MODIFIED get_batch ---

# --- MODIFIED estimate_loss to include test_loss and accept test_data ---
@torch.no_grad()
def estimate_loss(model, config, data_train, data_val, data_test):
    """ Estimate loss and metrics on train/val/test splits """
    out = {'loss': {}}
    eval_iters = config["eval_iters"]
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    n_layer = config["n_layer"]
    # num_experts = config["num_experts"] # num_experts not directly used in this function's logic for loss
    device = next(model.parameters()).device  # Get device from model

    model.eval()  # Set model to evaluation mode

    # Initialize accumulator for MoE metrics (only for validation split)
    val_metrics_accumulated = {
        f'L{layer_idx}': {'imbalance': 0.0, 'concentration': 0.0, 'utilization': 0.0,
                          'tpe_cv': 0.0, 'avg_prob_cv': 0.0, 'count': 0}
        for layer_idx in range(n_layer)
    }

    for split in ['train', 'val', 'test']:  # Include 'test'
        losses = torch.zeros(eval_iters)

        # Reset metrics accumulation for 'val' split if it's being processed
        if split == 'val':
            for layer_idx in range(n_layer):
                layer_metrics = val_metrics_accumulated[f'L{layer_idx}']
                for key in layer_metrics:  # Reset all numeric metrics, keep count at 0
                    layer_metrics[key] = 0.0 if key != 'count' else 0

        for k in range(eval_iters):
            # Pass all data splits to get_batch
            xb, yb = get_batch(split, data_train, data_val, data_test, block_size, batch_size, device)
            if xb is None or yb is None:  # Handle case where batch generation failed
                losses[k] = float('nan')  # Record NaN if batch is invalid
                continue

            logits, loss = model(xb, yb)
            losses[k] = loss.item() if loss is not None else float('nan')

            # Accumulate MoE metrics only for the 'val' split
            if split == 'val' and hasattr(model, 'blocks') and isinstance(model.blocks, nn.Sequential):
                for i, block in enumerate(model.blocks):  # Iterate through Blocks
                    if hasattr(block, 'sparse_moe') and isinstance(block.sparse_moe, SparseMoE):
                        smoe_layer = block.sparse_moe
                        layer_key = f'L{i}'
                        layer_metrics = val_metrics_accumulated[layer_key]

                        # Accumulate metrics only if they are valid numbers (not NaN)
                        if hasattr(smoe_layer, 'latest_imbalance'):  # Check if attributes exist
                            if not np.isnan(smoe_layer.latest_imbalance): layer_metrics[
                                'imbalance'] += smoe_layer.latest_imbalance
                            if not np.isnan(smoe_layer.latest_concentration): layer_metrics[
                                'concentration'] += smoe_layer.latest_concentration
                            if not np.isnan(smoe_layer.latest_utilization): layer_metrics[
                                'utilization'] += smoe_layer.latest_utilization
                            if not np.isnan(smoe_layer.latest_tpe_cv): layer_metrics[
                                'tpe_cv'] += smoe_layer.latest_tpe_cv
                            if not np.isnan(smoe_layer.latest_avg_prob_cv): layer_metrics[
                                'avg_prob_cv'] += smoe_layer.latest_avg_prob_cv
                            layer_metrics['count'] += 1  # Increment count only if metrics were added

        # Calculate mean loss for the current split, handling potential NaNs
        out['loss'][split] = np.nanmean(losses.numpy()) if not torch.all(torch.isnan(losses)) else float('nan')

    # Calculate overall average MoE metrics from the validation split
    total_metrics_sum = {'imbalance': 0.0, 'concentration': 0.0, 'utilization': 0.0,
                         'tpe_cv': 0.0, 'avg_prob_cv': 0.0}
    total_layers_with_metrics = 0  # Count layers that actually had metric updates

    for i in range(n_layer):
        count = val_metrics_accumulated[f'L{i}']['count']
        if count > 0:  # Only process layers where metrics were accumulated
            total_layers_with_metrics += 1
            metrics_sum = val_metrics_accumulated[f'L{i}']
            for key in total_metrics_sum:  # Iterate through the metric types
                metric_value = metrics_sum[key]
                if not np.isnan(metric_value):  # Ensure value is not NaN before division
                    avg_layer_metric = metric_value / count
                    total_metrics_sum[key] += avg_layer_metric  # Sum up the averages

    overall_avg_metrics = {}
    if total_layers_with_metrics > 0:
        for key in total_metrics_sum:
            overall_avg_metrics[key] = total_metrics_sum[key] / total_layers_with_metrics
    else:  # If no layers had metrics (e.g., count was 0 for all)
        for key in total_metrics_sum:  # Default to 0.0
            overall_avg_metrics[key] = 0.0

    model.train()  # Set model back to training mode
    return out['loss'], overall_avg_metrics  # Return dict of losses and dict of avg MoE metrics


# --- END MODIFIED estimate_loss ---

# ============================================================
# --- SECTION 1: Hyperparameter Sweep ---
# ============================================================
custom_print(f"\n--- Running Hyperparameter Sweep ---")

# Use config_sweep for sweep, or adjust if you want to sweep parameters from config_full
config_to_run_sweep_with = config_sweep  # Or config_full if sweeping its params
# Update eval_iters for the chosen config
config_to_run_sweep_with["eval_iters"] = (config_to_run_sweep_with["max_iters"] // config_to_run_sweep_with[
    "eval_interval"]) * config_to_run_sweep_with["eval_iters_scale_factor"]

alphas_to_sweep = [0.5, 1, 1.5, 2.0, 2.5]  # Example, adjust as needed
temperatures_to_sweep = [0.5, 1.0, 10.0]  # Example, adjust as needed
num_seeds_per_combo = 3  # Number of seeds to run per hyperparameter combination
base_seed_value = config_to_run_sweep_with["seed"]  # Use seed from chosen config

custom_print(f"Sweep Config Being Used: {config_to_run_sweep_with}")
custom_print(f"Sweeping Alphas: {alphas_to_sweep}")
custom_print(f"Sweeping Temperatures: {temperatures_to_sweep}")
custom_print(f"Running {num_seeds_per_combo} seeds per combination.")

# Use the globally defined train, val, test data for the sweep
sweep_train_data = train_data
sweep_val_data = val_data
sweep_test_data = test_data  # Added for test set

# --- MODIFIED DataFrame dtypes to include loss_test ---
dtypes_sweep = {'model_key': str, 'alpha': float, 'temperature': float, 'seed': int, 'iter': int,
                'loss_train': float, 'loss_val': float, 'loss_test': float,  # Added loss_test
                'Expert Utilization': float, 'Load Imbalance Ratio': float, 'Routing Concentration': float,
                'Expert Load Variation': float, 'Routing Probability CV': float}
df_sweep_results = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes_sweep.items()})
# --- END MODIFIED DataFrame dtypes ---

sweep_start_time_global = time.time()

# --- Sweep Loops (with Multi-Seed) ---
for alpha_val_sweep in alphas_to_sweep:
    for temp_val_sweep in temperatures_to_sweep:
        for seed_run_idx_sweep in range(num_seeds_per_combo):  # Iterate through seeds
            current_run_seed_sweep = base_seed_value + seed_run_idx_sweep
            run_key_sweep = f"alpha_{alpha_val_sweep}_temp_{temp_val_sweep}_seed_{current_run_seed_sweep}"

            # Check if results for this run_key (at least a completed marker) already exist
            completion_marker_file_pattern = f"./output/*{run_key_sweep}_Completed_*.csv_marker"  # Updated marker name
            running_marker_file = f"./output/{run_key_sweep}_Running_{str(int(time.time()))}.marker"

            if len(glob.glob(completion_marker_file_pattern)) > 0:
                custom_print(
                    f"Skipping already completed run: {run_key_sweep} (found marker: {completion_marker_file_pattern})")
                continue
            else:
                os.makedirs("./output", exist_ok=True)
                try:
                    with open(running_marker_file, 'w') as f_marker:
                        f_marker.write(f"Started at {datetime.now()}")
                except Exception as e_marker:
                    custom_print(f"Warning: Could not create running marker file {running_marker_file}: {e_marker}")

            custom_print(f"\n----- Starting Sweep Run: {run_key_sweep} -----")

            custom_print(f"Setting seed for this run: {current_run_seed_sweep}")
            torch.manual_seed(current_run_seed_sweep)
            np.random.seed(current_run_seed_sweep)

            # Update config for the current sweep iteration (use a copy to avoid modifying original config_to_run_sweep_with)
            current_iter_config = config_to_run_sweep_with.copy()
            current_iter_config['router_alpha'] = alpha_val_sweep
            current_iter_config['router_temperature'] = temp_val_sweep

            # --- Re-initialize Model and Optimizer for each seed/run ---
            model = SparseMoELanguageModel(
                vocab_size=full_vocab_size,
                n_embed=current_iter_config["n_embed"],
                n_head=current_iter_config["n_head"],
                n_layer=current_iter_config["n_layer"],
                num_experts=current_iter_config["num_experts"],
                block_size=current_iter_config["block_size"],
                dropout=current_iter_config["dropout"],
                router_alpha=current_iter_config["router_alpha"],
                router_temperature=current_iter_config["router_temperature"]
            ).to(device)
            custom_print(f"Sweep Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

            optimizer = torch.optim.AdamW(model.parameters(), lr=current_iter_config["learning_rate"])

            run_specific_start_time = time.time()
            # --- Training Loop ---
            for iter_num_sweep in range(current_iter_config["max_iters"]):
                # Evaluate at intervals or at the last iteration
                if iter_num_sweep % current_iter_config["eval_interval"] == 0 or iter_num_sweep == current_iter_config[
                    "max_iters"] - 1:
                    # Pass all three data splits to estimate_loss
                    losses_dict_sweep, avg_metrics_val_sweep = estimate_loss(model, current_iter_config,
                                                                             sweep_train_data, sweep_val_data,
                                                                             sweep_test_data)

                    loss_train_val_sweep = losses_dict_sweep.get('train', float('nan'))
                    loss_val_val_sweep = losses_dict_sweep.get('val', float('nan'))
                    loss_test_val_sweep = losses_dict_sweep.get('test', float('nan'))  # Get test loss

                    # --- MODIFIED new_row_data to include loss_test ---
                    new_row_data_sweep = {
                        'model_key': f"alpha_{alpha_val_sweep}_temp_{temp_val_sweep}",
                        'alpha': alpha_val_sweep, 'temperature': temp_val_sweep, 'seed': current_run_seed_sweep,
                        'iter': iter_num_sweep,
                        'loss_train': loss_train_val_sweep, 'loss_val': loss_val_val_sweep,
                        'loss_test': loss_test_val_sweep,
                        'Expert Utilization': avg_metrics_val_sweep.get('utilization', 0.0),
                        'Load Imbalance Ratio': avg_metrics_val_sweep.get('imbalance', 0.0),
                        'Routing Concentration': avg_metrics_val_sweep.get('concentration', 0.0),
                        'Expert Load Variation': avg_metrics_val_sweep.get('tpe_cv', 0.0),
                        'Routing Probability CV': avg_metrics_val_sweep.get('avg_prob_cv', 0.0)
                    }
                    new_row_df_sweep = pd.DataFrame([new_row_data_sweep])
                    if not new_row_df_sweep.empty and not new_row_df_sweep.isnull().all().all():
                        df_sweep_results = pd.concat([df_sweep_results, new_row_df_sweep], ignore_index=True)
                    # --- END MODIFIED new_row_data ---

                    # --- MODIFIED print statement to include Test Loss ---
                    custom_print(
                        f"Sweep Run: {run_key_sweep} | Iter: {iter_num_sweep:<4}/{current_iter_config['max_iters']:<4} || "
                        f"Train Loss: {loss_train_val_sweep:.4f} | Val Loss: {loss_val_val_sweep:.4f} | Test Loss: {loss_test_val_sweep:.4f}")
                    custom_print(
                        f"Val Metrics (Avg) || Util: {avg_metrics_val_sweep.get('utilization', 0.0):<6.2f} | Imb: {avg_metrics_val_sweep.get('imbalance', 0.0):<6.2f} | Conc: {avg_metrics_val_sweep.get('concentration', 0.0):<6.2f} | TPE_CV: {avg_metrics_val_sweep.get('tpe_cv', 0.0):<6.2f} | Pr_CV: {avg_metrics_val_sweep.get('avg_prob_cv', 0.0):<6.2f}")
                    # --- END MODIFIED print statement ---

                # Training step
                xb_sweep, yb_sweep = get_batch('train', sweep_train_data, sweep_val_data, sweep_test_data,
                                               current_iter_config["block_size"], current_iter_config["batch_size"],
                                               device)
                if xb_sweep is None or yb_sweep is None: continue

                logits_sweep, loss_sweep = model(xb_sweep, yb_sweep)
                optimizer.zero_grad(set_to_none=True)
                loss_sweep.backward()
                optimizer.step()

                if iter_num_sweep > 0 and iter_num_sweep % (current_iter_config["eval_interval"] * 5) == 0:
                    interim_results_file = f"./output/sweep_results_interim_{str(int(sweep_start_time_global))}.csv"
                    try:
                        df_sweep_results.to_csv(interim_results_file, index=False)
                        custom_print(f"Saved interim results to {interim_results_file} at iter {iter_num_sweep}")
                    except Exception as e_csv:
                        custom_print(f"Warning: Could not save interim CSV {interim_results_file}: {e_csv}")

            completed_marker_file = f"./output/{run_key_sweep}_Completed_{str(int(time.time()))}.csv_marker"
            try:
                with open(completed_marker_file, 'w') as f_marker:
                    f_marker.write(f"Completed at {datetime.now()}")
                if os.path.exists(running_marker_file):
                    os.remove(running_marker_file)
            except Exception as e_marker:
                custom_print(
                    f"Warning: Could not create/update completion marker file {completed_marker_file}: {e_marker}")

            run_specific_end_time = time.time()
            model_save_path = f"./models_out/model_{run_key_sweep}_{str(int(run_specific_start_time))}.pth"
            torch.save(model.state_dict(), model_save_path)
            custom_print(
                f"----- Sweep Run {run_key_sweep} completed in {run_specific_end_time - run_specific_start_time:.2f} seconds -----")

sweep_end_time_global = time.time()
custom_print(
    f"\n--- Full Sweep (all seeds and configurations) completed in {sweep_end_time_global - sweep_start_time_global:.2f} seconds ---")

final_sweep_results_file = f'./output/sweep_results_all_seeds_with_test_best_val_{str(int(sweep_start_time_global))}.csv'  # Updated filename
custom_print(f"\nSaving final detailed sweep results to {final_sweep_results_file}")
try:
    if not df_sweep_results.empty:
        df_sweep_results.to_csv(final_sweep_results_file, index=False)
    else:
        custom_print("Warning: Sweep results DataFrame is empty. Nothing to save.")
except Exception as e_csv:
    custom_print(f"Error saving final CSV {final_sweep_results_file}: {e_csv}")

custom_print(
    "\n--- Aggregated Sweep Results (Based on Best Validation Iteration per Run, Averaged Across Seeds) ---")  # Updated Title

if not df_sweep_results.empty:
    # --- Select rows corresponding to the minimum validation loss for each run (alpha, temp, seed) ---
    idx_best_val_iter = df_sweep_results.groupby(['alpha', 'temperature', 'seed'])['loss_val'].idxmin()
    df_best_val_iter_runs = df_sweep_results.loc[idx_best_val_iter].copy()
    # --- END MODIFICATION ---

    metric_cols_to_aggregate_best_val = ['iter', 'loss_train', 'loss_val', 'loss_test',
                                         'Expert Utilization', 'Load Imbalance Ratio',
                                         'Routing Concentration', 'Expert Load Variation', 'Routing Probability CV']

    best_sweep_params_overall = None
    if not df_best_val_iter_runs.empty:
        # Group by alpha and temperature, then aggregate metrics from the best validation iteration runs
        grouped_best_val_results = df_best_val_iter_runs.groupby(['alpha', 'temperature'])
        aggregated_best_val_stats = grouped_best_val_results[metric_cols_to_aggregate_best_val].agg(['mean', 'std'])
        aggregated_best_val_stats.columns = ['_'.join(col).strip() for col in aggregated_best_val_stats.columns.values]

        if 'loss_val_mean' in aggregated_best_val_stats.columns:
            aggregated_best_val_stats = aggregated_best_val_stats.sort_values('loss_val_mean')
        else:
            custom_print("Warning: 'loss_val_mean' column not found for sorting aggregated results.")

        custom_print("\nAggregated Statistics (from Best Validation Iteration, Mean and Std across seeds):")
        custom_print(aggregated_best_val_stats)

        agg_best_val_results_file = f'./output/sweep_results_aggregated_best_val_with_test_{str(int(sweep_start_time_global))}.csv'  # Updated filename
        custom_print(f"\nSaving aggregated best validation iteration results to {agg_best_val_results_file}")
        try:
            aggregated_best_val_stats.to_csv(agg_best_val_results_file)
        except Exception as e_csv:
            custom_print(f"Error saving aggregated CSV {agg_best_val_results_file}: {e_csv}")

        if 'loss_val_mean' in aggregated_best_val_stats.columns and not aggregated_best_val_stats.empty:
            best_params_idx_agg = aggregated_best_val_stats['loss_val_mean'].idxmin()
            best_sweep_params_overall = {'alpha': best_params_idx_agg[0], 'temperature': best_params_idx_agg[1]}
            custom_print(
                f"\nBest parameters found from sweep (based on min loss_val_mean from best iteration): Alpha={best_sweep_params_overall['alpha']}, Temperature={best_sweep_params_overall['temperature']}")

            if 'loss_test_mean' in aggregated_best_val_stats.columns and 'loss_test_std' in aggregated_best_val_stats.columns:
                custom_print(
                    f"Corresponding mean test loss (at best val iter): {aggregated_best_val_stats.loc[best_params_idx_agg, 'loss_test_mean']:.4f} "
                    f"(std: {aggregated_best_val_stats.loc[best_params_idx_agg, 'loss_test_std']:.4f})")
                custom_print(
                    f"Achieved at mean iteration: {aggregated_best_val_stats.loc[best_params_idx_agg, 'iter_mean']:.0f}")

            else:
                custom_print("Test loss mean/std not available for the best parameters in aggregated_best_val_stats.")
        else:
            custom_print("Could not determine best parameters: 'loss_val_mean' not in columns or no data.")
    else:
        custom_print("No data after selecting for best validation iteration; cannot aggregate.")
else:
    custom_print("Sweep results DataFrame is empty. Cannot perform aggregation.")

if log_file:
    try:
        log_file.close()
    except Exception as e:
        print(f"Error closing log file: {e}")

print("\n--- Experiment Script Finished ---")
