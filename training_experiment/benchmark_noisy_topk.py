# benchmark_noisy_topk.py
import os
import time
import glob

# --- Imports ---
import numpy as np
import pandas as pd
# import seaborn as sns # Optional: if you plan to add plotting directly in this script
import torch
import torch.nn as nn
from torch.nn import functional as F

from datetime import datetime

# --- Logging Setup ---
log_file_path = "benchmark_noisy_topk_log.txt"
try:
    log_file = open(log_file_path, "a")
except Exception as e:
    print(f"Error opening log file {log_file_path}: {e}")
    log_file = None


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


# --- Base Configuration for NoisyTopkRouter Benchmarking ---
config_base = {
    "learning_rate": 1e-3,
    "dropout": 0.1,
    "eval_iters_scale_factor": 10,  # Multiplies (max_iters // eval_interval)
    # router_top_k will be set by the sweep
    "seed": 42,
    # General model parameters (can be overridden by specific configs like sweep/full)
    "batch_size": 128,  # Example full run, was 64 for sweep
    "block_size": 128,  # Example full run, was 64 for sweep
    "max_iters": 5000,  # Full run iterations
    "eval_interval": 100,  # Full run eval interval
    "n_embed": 128,  # Example full run
    "n_head": 8,  # Example full run
    "n_layer": 6,  # Example full run
    "num_experts": 8,  # Example full run
}

# Use the base config, can be overridden if you define other configs like config_full
current_config_params = config_base.copy()
current_config_params["eval_iters"] = \
    (current_config_params["max_iters"] // current_config_params["eval_interval"]) * \
    current_config_params["eval_iters_scale_factor"]

custom_print(f"Using base configuration for NoisyTopkRouter benchmark: {current_config_params}")

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else \
    ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
custom_print(f"Using device: {device}")

# --- Data Loading and Preprocessing (Tiny Shakespeare) ---
data_file = 'input.txt'  # Ensure this file is in the same directory
if not os.path.exists(data_file):
    custom_print(f"Error: Dataset file '{data_file}' not found!")
    custom_print("Please download Tiny Shakespeare input.txt and place it in the script's directory.")
    exit()

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

    # Data Split: 80% train, 10% validation, 10% test
    n = len(data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_data = data[:n_train]
    val_data = data[n_train: n_train + n_val]
    test_data = data[n_train + n_val:]
    custom_print(
        f"Data Split: Train tokens: {len(train_data)}, Valid tokens: {len(val_data)}, Test tokens: {len(test_data)}")

except Exception as e:
    custom_print(f"Error processing dataset file: {e}")
    exit()


# --- Model Definitions ---
class Head(nn.Module):
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
        k = self.key(x);
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x);
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
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
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
        custom_print(f"Initialized NoisyTopkRouter with top_k = {self.top_k}, num_experts = {num_experts}")
        if self.top_k > num_experts:
            custom_print(f"Warning: top_k ({self.top_k}) > num_experts ({num_experts}). Setting top_k to num_experts.")
            self.top_k = num_experts

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        # Ensure top_k is not greater than the number of experts
        current_top_k = min(self.top_k, noisy_logits.size(-1))
        if current_top_k == 0 and noisy_logits.size(-1) > 0:  # Cannot take top 0
            current_top_k = 1
        elif noisy_logits.size(-1) == 0:  # No experts to choose from
            # This case should ideally not happen if num_experts > 0
            # Return a uniform distribution or handle error appropriately
            # For now, let's assume num_experts > 0
            # If noisy_logits is empty, this will error. Add a check for num_experts in __init__
            return torch.zeros_like(noisy_logits), None  # Or handle as an error state

        top_k_logits, indices = noisy_logits.topk(current_top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output_probs = F.softmax(sparse_logits, dim=-1)
        return router_output_probs, indices


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, dropout, router_top_k):
        super().__init__()
        self.num_experts = num_experts
        if num_experts <= 0:
            custom_print(f"Error: num_experts must be positive, got {num_experts}")
            raise ValueError("num_experts must be positive for SparseMoE")
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k=router_top_k)
        self.experts = nn.ModuleList([Expert(n_embed, dropout) for _ in range(num_experts)])
        self.register_buffer('latest_tpe', torch.zeros(num_experts, dtype=torch.float32))
        self.latest_imbalance: float = 1.0
        self.latest_concentration: float = 0.0
        self.latest_utilization: float = 0.0
        self.latest_tpe_cv: float = 0.0
        self.latest_avg_prob_cv: float = 0.0
        self.activation_threshold = 1e-9

    def calculate_metrics(self, gating_output_no_grad):
        num_tokens = gating_output_no_grad.shape[0]
        if num_tokens == 0 or self.num_experts == 0:
            self.latest_tpe.zero_();
            self.latest_imbalance = 1.0;
            self.latest_concentration = 0.0
            self.latest_utilization = 0.0;
            self.latest_tpe_cv = 0.0;
            self.latest_avg_prob_cv = 0.0
            return
        with torch.no_grad():
            is_active = gating_output_no_grad > self.activation_threshold
            tpe = is_active.sum(dim=0).float()
            if self.latest_tpe.shape[0] == tpe.shape[0]:
                self.latest_tpe.copy_(tpe)
            else:
                self.register_buffer('latest_tpe', tpe.clone())
            if self.num_experts > 1:
                mean_tpe = tpe.mean();
                std_tpe = tpe.std(unbiased=False)
                self.latest_tpe_cv = (std_tpe / (mean_tpe + 1e-9)).item() if mean_tpe > 1e-9 else 0.0
            else:
                self.latest_tpe_cv = 0.0
            mean_tpe_val = tpe.mean().item()
            self.latest_imbalance = tpe.max().item() / (mean_tpe_val + 1e-9) if mean_tpe_val > 0 else 1.0
            max_p_per_token, _ = gating_output_no_grad.max(dim=-1)
            self.latest_concentration = max_p_per_token.mean().item()
            num_active_experts = (tpe > 0).sum().item()
            self.latest_utilization = num_active_experts / self.num_experts if self.num_experts > 0 else 0.0
            if self.num_experts > 1:
                avg_prob_per_expert = gating_output_no_grad.mean(dim=0);
                mean_avg_prob = avg_prob_per_expert.mean();
                std_avg_prob = avg_prob_per_expert.std(unbiased=False)
                self.latest_avg_prob_cv = (
                        std_avg_prob / (mean_avg_prob + 1e-9)).item() if mean_avg_prob > 1e-9 else 0.0
            else:
                self.latest_avg_prob_cv = 0.0

    def forward(self, x):
        batch_size, seq_len, n_embed = x.shape
        num_tokens = batch_size * seq_len
        x_reshaped = x.view(num_tokens, n_embed)

        gating_probabilities, _ = self.router(x_reshaped)  # Use probabilities from NoisyTopkRouter

        self.calculate_metrics(gating_probabilities.detach())
        final_output = torch.zeros_like(x_reshaped)
        for i, expert_module in enumerate(self.experts):
            token_indices = torch.nonzero(gating_probabilities[:, i] > self.activation_threshold).squeeze(-1)
            if token_indices.numel() == 0: continue
            expert_input_tokens = x_reshaped[token_indices]
            active_gating_scores_for_expert = gating_probabilities[token_indices, i].unsqueeze(1)
            expert_output_tokens = expert_module(expert_input_tokens)
            weighted_expert_output = expert_output_tokens * active_gating_scores_for_expert
            final_output.index_add_(0, token_indices, weighted_expert_output)
        final_output = final_output.view(batch_size, seq_len, n_embed)
        return final_output


class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, block_size, dropout, router_top_k):
        super().__init__()
        assert n_embed % n_head == 0
        self.self_attn = MultiHeadAttention(n_embed, n_head, block_size, dropout)
        self.sparse_moe = SparseMoE(n_embed, num_experts, dropout, router_top_k=router_top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.sparse_moe(self.ln2(x))
        return x


class LanguageModelWithNoisyTopK(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, num_experts, block_size, dropout, router_top_k):
        super().__init__()
        self.n_embed = n_embed;
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed=n_embed, n_head=n_head, num_experts=num_experts,
                  block_size=block_size, dropout=dropout, router_top_k=router_top_k)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed);
        self.lm_head = nn.Linear(n_embed, vocab_size)
        custom_print(f"Initialized LanguageModel with NoisyTopkRouter (top_k={router_top_k}) and {n_layer} layers.")

    def forward(self, idx, targets=None):
        B, T = idx.shape;
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_indices = torch.arange(T, device=device)

        current_T_for_pos = T
        if T > self.block_size:
            pos_indices_for_emb = pos_indices[-self.block_size:]
            tok_emb_for_sum = tok_emb[:, -self.block_size:, :]
            current_T_for_pos = self.block_size
        elif T < self.block_size:
            pos_indices_for_emb = pos_indices[:T]
            tok_emb_for_sum = tok_emb
        else:  # T == self.block_size
            pos_indices_for_emb = pos_indices
            tok_emb_for_sum = tok_emb

        pos_emb = self.position_embedding_table(pos_indices_for_emb)

        # Ensure dimensions match for addition if T was dynamic
        if tok_emb_for_sum.shape[1] != pos_emb.shape[0]:
            # This might happen if pos_emb was not sliced correctly when T < block_size
            # or if tok_emb was not handled when T > block_size for sum
            # For safety, try to align pos_emb to tok_emb_for_sum's sequence length
            if pos_emb.shape[0] > tok_emb_for_sum.shape[1]:
                pos_emb = pos_emb[:tok_emb_for_sum.shape[1]]
            # If pos_emb is shorter, this implies an issue with tok_emb_for_sum logic or input idx
            # This section requires careful handling of sequence lengths.
            # Assuming T of tok_emb_for_sum is the definitive sequence length for this step.

        x = tok_emb_for_sum + pos_emb
        x = self.blocks(x);
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            T_processed = x.shape[1]
            targets_for_loss = targets
            if targets.shape[1] > T_processed:
                targets_for_loss = targets[:, -T_processed:]
            elif targets.shape[1] < T_processed:
                custom_print(f"Warning: Targets len {targets.shape[1]} < processed logits len {T_processed}.")

            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets_for_loss.reshape(-1)
            if logits_flat.shape[0] != targets_flat.shape[0]:
                custom_print(
                    f"ERROR: Mismatch in flattened logits ({logits_flat.shape[0]}) and targets ({targets_flat.shape[0]}) for loss.")
                min_len = min(logits_flat.shape[0], targets_flat.shape[0])
                loss = F.cross_entropy(logits_flat[:min_len], targets_flat[:min_len]) if min_len > 0 else torch.tensor(
                    float('nan'), device=device)
            elif logits_flat.shape[0] == 0:  # No elements to compute loss on
                loss = torch.tensor(float('nan'), device=device)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss


def get_batch(split, data_source_train, data_source_val, data_source_test, block_size, batch_size, device_type):
    data_src = data_source_train if split == 'train' else \
        (data_source_val if split == 'val' else data_source_test)
    if len(data_src) < block_size + 1:
        # custom_print(f"Warning: Dataset split '{split}' (len: {len(data_src)}) is too small. Skipping batch.")
        return None, None
    max_start_idx = len(data_src) - block_size - 1
    ix = torch.randint(max_start_idx + 1, (batch_size,))
    x_batch = torch.stack([data_src[i: i + block_size] for i in ix])
    y_batch = torch.stack([data_src[i + 1: i + block_size + 1] for i in ix])
    x_batch, y_batch = x_batch.to(device_type), y_batch.to(device_type)
    return x_batch, y_batch


@torch.no_grad()
def estimate_loss(model_to_eval, config_eval, data_src_train, data_src_val, data_src_test):
    out_losses = {'loss': {}}
    eval_iters_count = config_eval["eval_iters"];
    b_size = config_eval["block_size"];
    batch_s = config_eval["batch_size"]
    n_l = config_eval["n_layer"];
    device = next(model_to_eval.parameters()).device
    model_to_eval.eval()
    val_metrics_agg = {f'L{layer_i}': {m_name: 0.0 for m_name in
                                       ['imbalance', 'concentration', 'utilization', 'tpe_cv', 'avg_prob_cv']} | {
                                          'count': 0} for layer_i in range(n_l)}
    for data_split_name in ['train', 'val', 'test']:
        split_losses = torch.zeros(eval_iters_count)
        if data_split_name == 'val':
            for layer_i in range(n_l):
                for metric_key in val_metrics_agg[f'L{layer_i}']: val_metrics_agg[f'L{layer_i}'][
                    metric_key] = 0.0 if metric_key != 'count' else 0
        for k_idx in range(eval_iters_count):
            xb_eval, yb_eval = get_batch(data_split_name, data_src_train, data_src_val, data_src_test, b_size, batch_s,
                                         device)
            if xb_eval is None or yb_eval is None: split_losses[k_idx] = float('nan'); continue
            _, current_loss = model_to_eval(xb_eval, yb_eval)
            split_losses[k_idx] = current_loss.item() if current_loss is not None and not torch.isnan(
                current_loss) else float('nan')
            if data_split_name == 'val' and hasattr(model_to_eval, 'blocks') and isinstance(model_to_eval.blocks,
                                                                                            nn.Sequential):
                for block_idx, model_block in enumerate(model_to_eval.blocks):
                    if hasattr(model_block, 'sparse_moe') and isinstance(model_block.sparse_moe, SparseMoE):
                        moe_l = model_block.sparse_moe;
                        current_layer_key = f'L{block_idx}';
                        metrics_for_layer = val_metrics_agg[current_layer_key]
                        if hasattr(moe_l, 'latest_imbalance'):
                            for m_name_val in ['imbalance', 'concentration', 'utilization', 'tpe_cv', 'avg_prob_cv']:
                                metric_value_val = getattr(moe_l, f"latest_{m_name_val}", float('nan'))
                                if not np.isnan(metric_value_val): metrics_for_layer[m_name_val] += metric_value_val
                            metrics_for_layer['count'] += 1
        current_split_loss_mean = np.nanmean(split_losses.numpy()) if not torch.all(
            torch.isnan(split_losses)) else float('nan')
        out_losses['loss'][data_split_name] = current_split_loss_mean

    avg_moe_metrics_overall = {m_key: 0.0 for m_key in
                               ['imbalance', 'concentration', 'utilization', 'tpe_cv', 'avg_prob_cv']}
    num_layers_with_valid_metrics = 0
    for layer_idx_val in range(n_l):
        metrics_count = val_metrics_agg[f'L{layer_idx_val}']['count']
        if metrics_count > 0:
            num_layers_with_valid_metrics += 1
            for m_key_val in avg_moe_metrics_overall:
                sum_metric_for_layer = val_metrics_agg[f'L{layer_idx_val}'][m_key_val]
                if not np.isnan(sum_metric_for_layer): avg_moe_metrics_overall[
                    m_key_val] += sum_metric_for_layer / metrics_count
    if num_layers_with_valid_metrics > 0:
        for m_key_val_avg in avg_moe_metrics_overall: avg_moe_metrics_overall[
            m_key_val_avg] /= num_layers_with_valid_metrics
    model_to_eval.train()
    return out_losses['loss'], avg_moe_metrics_overall


# ============================================================
# --- SECTION: NoisyTopkRouter Benchmark Sweep ---
# ============================================================
custom_print(f"\n--- Running NoisyTopkRouter Benchmark Sweep ---")

top_k_values_to_sweep = [1, 2, 3]  # MODIFIED as per user request
num_seeds_per_combo = 1  # Example: Number of seeds per top_k value
base_seed_value = current_config_params["seed"]

config_for_sweep = current_config_params.copy()
custom_print(f"Base Config for Sweep: {config_for_sweep}")
custom_print(f"Sweeping Top_K Values: {top_k_values_to_sweep}")
custom_print(f"Running {num_seeds_per_combo} seeds per top_k configuration.")

dtypes_benchmark = {'model_key': str, 'top_k': int, 'seed': int, 'iter': int,
                    'loss_train': float, 'loss_val': float, 'loss_test': float,
                    'Expert Utilization': float, 'Load Imbalance Ratio': float,
                    'Routing Concentration': float, 'Expert Load Variation': float,
                    'Routing Probability CV': float}
df_benchmark_results = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes_benchmark.items()})

benchmark_start_time_global = time.time()
os.makedirs("./output", exist_ok=True)

for top_k_val in top_k_values_to_sweep:
    if top_k_val > config_for_sweep["num_experts"]:
        custom_print(f"Skipping top_k={top_k_val} as it's greater than num_experts={config_for_sweep['num_experts']}")
        continue
    if top_k_val <= 0:
        custom_print(f"Skipping invalid top_k={top_k_val}. Must be > 0.")
        continue

    for seed_idx in range(num_seeds_per_combo):
        current_seed = base_seed_value + seed_idx
        run_key = f"noisy_topk_{top_k_val}_seed_{current_seed}"

        completion_marker_pattern = f"./output/{run_key}_Completed_*.marker"  # Exact match
        running_marker_file = f"./output/{run_key}_Running_{str(int(time.time()))}.marker"
        if glob.glob(completion_marker_pattern):  # Use glob.glob for pattern matching
            custom_print(f"Skipping already completed run: {run_key}")
            continue
        try:
            with open(running_marker_file, 'w') as f_marker:
                f_marker.write(f"Started at {datetime.now()}")
        except Exception as e_marker:
            custom_print(f"Warning: Could not create running marker: {e_marker}")

        custom_print(f"\n----- Starting Benchmark Run: {run_key} -----")
        torch.manual_seed(current_seed);
        np.random.seed(current_seed)

        iter_config = config_for_sweep.copy()
        iter_config['router_top_k'] = top_k_val

        model = LanguageModelWithNoisyTopK(
            vocab_size=full_vocab_size,
            n_embed=iter_config["n_embed"], n_head=iter_config["n_head"],
            n_layer=iter_config["n_layer"], num_experts=iter_config["num_experts"],
            block_size=iter_config["block_size"], dropout=iter_config["dropout"],
            router_top_k=iter_config["router_top_k"]
        ).to(device)
        custom_print(f"Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
        optimizer = torch.optim.AdamW(model.parameters(), lr=iter_config["learning_rate"])

        run_start_time_current = time.time()
        for iter_num in range(iter_config["max_iters"]):
            if iter_num % iter_config["eval_interval"] == 0 or iter_num == iter_config["max_iters"] - 1:
                losses_at_iter, metrics_at_iter = estimate_loss(model, iter_config, train_data, val_data, test_data)
                current_row_data = {
                    'model_key': f"noisy_topk_{top_k_val}", 'top_k': top_k_val, 'seed': current_seed, 'iter': iter_num,
                    'loss_train': losses_at_iter.get('train', float('nan')),
                    'loss_val': losses_at_iter.get('val', float('nan')),
                    'loss_test': losses_at_iter.get('test', float('nan')),
                    'Expert Utilization': metrics_at_iter.get('utilization', 0.0),
                    'Load Imbalance Ratio': metrics_at_iter.get('imbalance', 0.0),
                    'Routing Concentration': metrics_at_iter.get('concentration', 0.0),
                    'Expert Load Variation': metrics_at_iter.get('tpe_cv', 0.0),
                    'Routing Probability CV': metrics_at_iter.get('avg_prob_cv', 0.0)
                }
                new_row_df = pd.DataFrame([current_row_data])
                if not new_row_df.empty and not new_row_df.isnull().all().all():
                    df_benchmark_results = pd.concat([df_benchmark_results, new_row_df], ignore_index=True)
                custom_print(
                    f"Run: {run_key} | Iter: {iter_num:<4}/{iter_config['max_iters']:<4} || Train: {current_row_data['loss_train']:.4f} | Val: {current_row_data['loss_val']:.4f} | Test: {current_row_data['loss_test']:.4f}")
                custom_print(
                    f"Val Metrics || Util: {current_row_data['Expert Utilization']:<6.2f} | Imb: {current_row_data['Load Imbalance Ratio']:<6.2f} | Conc: {current_row_data['Routing Concentration']:<6.2f} | TPE_CV: {current_row_data['Expert Load Variation']:<6.2f} | Pr_CV: {current_row_data['Routing Probability CV']:<6.2f}")

            xb_train, yb_train = get_batch('train', train_data, val_data, test_data, iter_config["block_size"],
                                           iter_config["batch_size"], device)
            if xb_train is None or yb_train is None: continue

            _, loss_train_step = model(xb_train, yb_train)
            if loss_train_step is not None and not torch.isnan(loss_train_step):
                optimizer.zero_grad(set_to_none=True);
                loss_train_step.backward();
                optimizer.step()
            else:
                custom_print(
                    f"Warning: NaN or None loss encountered at iter {iter_num} for run {run_key}. Skipping optimizer step.")

            if iter_num > 0 and iter_num % (iter_config["eval_interval"] * 10) == 0:
                interim_file_path = f"./output/benchmark_noisy_topk_interim_{str(int(benchmark_start_time_global))}.csv"
                try:
                    if not df_benchmark_results.empty: df_benchmark_results.to_csv(interim_file_path, index=False)
                    # custom_print(f"Saved interim results to {interim_file_path} at iter {iter_num}") # Less verbose
                except Exception as e_csv_interim:
                    custom_print(f"Warning: Could not save interim CSV: {e_csv_interim}")

        completed_marker_path = f"./output/{run_key}_Completed_{str(int(time.time()))}.marker"
        try:
            with open(completed_marker_path, 'w') as f_m_comp:
                f_m_comp.write(f"Completed at {datetime.now()}")
            if os.path.exists(running_marker_file): os.remove(running_marker_file)
        except Exception as e_m_comp:
            custom_print(f"Warning: Could not manage marker files: {e_m_comp}")
        custom_print(
            f"----- Benchmark Run {run_key} completed in {time.time() - run_start_time_current:.2f} seconds -----")

benchmark_end_time_global = time.time()
custom_print(
    f"\n--- Full NoisyTopkRouter Benchmark completed in {benchmark_end_time_global - benchmark_start_time_global:.2f} seconds ---")

final_benchmark_results_file_path = f'./output/benchmark_noisy_topk_all_runs_{str(int(benchmark_start_time_global))}.csv'
custom_print(f"\nSaving final detailed benchmark results to {final_benchmark_results_file_path}")
try:
    if not df_benchmark_results.empty:
        df_benchmark_results.to_csv(final_benchmark_results_file_path, index=False)
    else:
        custom_print("Warning: Benchmark results DataFrame is empty.")
except Exception as e_csv_final:
    custom_print(f"Error saving final CSV: {e_csv_final}")

custom_print("\n--- Aggregated Benchmark Results (Based on Best Validation Iteration per Run) ---")
if not df_benchmark_results.empty:
    for col_name in ['top_k', 'loss_val', 'iter', 'loss_train', 'loss_test',
                     'Expert Utilization', 'Load Imbalance Ratio', 'Routing Concentration',
                     'Expert Load Variation', 'Routing Probability CV']:
        if col_name in df_benchmark_results.columns:
            df_benchmark_results[col_name] = pd.to_numeric(df_benchmark_results[col_name], errors='coerce')
    df_benchmark_results.dropna(subset=['loss_val', 'top_k', 'seed'], inplace=True)

    if not df_benchmark_results.empty:
        idx_best_val_iteration = df_benchmark_results.groupby(['top_k', 'seed'])['loss_val'].idxmin()
        df_best_val_iteration_runs = df_benchmark_results.loc[idx_best_val_iteration].copy()

        metric_cols_for_aggregation = ['iter', 'loss_train', 'loss_val', 'loss_test',
                                       'Expert Utilization', 'Load Imbalance Ratio', 'Routing Concentration',
                                       'Expert Load Variation', 'Routing Probability CV']
        if not df_best_val_iteration_runs.empty:
            df_best_val_iteration_runs.dropna(subset=['top_k'], inplace=True)
            if not df_best_val_iteration_runs.empty:
                grouped_final_results = df_best_val_iteration_runs.groupby(['top_k'])
                aggregated_final_stats = grouped_final_results[metric_cols_for_aggregation].agg(['mean', 'std'])
                aggregated_final_stats.columns = ['_'.join(col_tuple).strip() for col_tuple in
                                                  aggregated_final_stats.columns.values]
                if 'loss_val_mean' in aggregated_final_stats.columns: aggregated_final_stats = aggregated_final_stats.sort_values(
                    'loss_val_mean')
                custom_print(
                    "\nAggregated Statistics for NoisyTopkRouter (from Best Validation Iteration, Averaged Across Seeds):")
                custom_print(aggregated_final_stats)
                aggregated_results_file_path = f'./output/benchmark_noisy_topk_aggregated_{str(int(benchmark_start_time_global))}.csv'
                custom_print(f"\nSaving aggregated benchmark results to {aggregated_results_file_path}")
                try:
                    aggregated_final_stats.to_csv(aggregated_results_file_path)
                except Exception as e_csv_agg:
                    custom_print(f"Error saving aggregated CSV: {e_csv_agg}")

                if 'loss_val_mean' in aggregated_final_stats.columns and not aggregated_final_stats.empty:
                    best_overall_top_k_idx = aggregated_final_stats['loss_val_mean'].idxmin()
                    best_overall_top_k = best_overall_top_k_idx
                    custom_print(f"\nBest top_k value found from benchmark: {best_overall_top_k}")
                    if 'loss_test_mean' in aggregated_final_stats.columns and \
                            'loss_test_std' in aggregated_final_stats.columns and \
                            best_overall_top_k_idx in aggregated_final_stats.index:
                        custom_print(
                            f"  Corresponding mean test loss: {aggregated_final_stats.loc[best_overall_top_k_idx, 'loss_test_mean']:.4f} "
                            f"(std: {aggregated_final_stats.loc[best_overall_top_k_idx, 'loss_test_std']:.4f})")
                        custom_print(
                            f"  Achieved at mean iteration: {aggregated_final_stats.loc[best_overall_top_k_idx, 'iter_mean']:.0f}")
                else:
                    custom_print("Could not determine the best top_k value.")
            else:
                custom_print("No data left for aggregation after dropping NaNs for grouping.")
        else:
            custom_print("No data found after selecting for best validation iteration.")
    else:
        custom_print("No data in df_benchmark_results after initial NaN drop.")
else:
    custom_print("Benchmark results DataFrame is empty.")

if log_file:
    try:
        log_file.close()
    except Exception as e_log_close:
        print(f"Error closing log file: {e_log_close}")
custom_print("\n--- NoisyTopkRouter Benchmark Script Finished ---")
