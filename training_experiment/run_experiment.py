import os
import time

# --- Imports ---
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import functional as F

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.8, rc={"figure.figsize": (8, 6)}, font="monospace",
              context="notebook", color_codes=True)
from datetime import datetime

log_file = open("experiment_log.txt", "a")


def custom_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    message = " ".join(str(a) for a in args)
    full_message = f"{timestamp} {message}"
    print(full_message, **kwargs)
    print(full_message, file=log_file)
    log_file.flush()


try:
    from entmax import entmax_bisect

    custom_print("Successfully imported entmax_bisect.")
    ENTMAX_AVAILABLE = True
except ImportError:
    custom_print("Warning: entmax library not found or import failed.")
    custom_print("         AlphaEntmaxRouter will default to Softmax for alpha != 1.0.")
    custom_print("         Install entmax: pip install entmax")
    ENTMAX_AVAILABLE = False


    def entmax_bisect(*args, **kwargs):
        custom_print("Fallback: Using Softmax instead of entmax_bisect (for alpha != 1.0).")
        logits = args[0]
        return F.softmax(logits, dim=-1)

# --- Base Configuration () ---
config_base = {
    "learning_rate": 1e-3,
    "dropout": 0.1,
    "eval_iters_scale_factor": 10,
    "router_alpha": 1.5,
    "router_temperature": 1.0,
    "entmax_n_iter": 25,
    "seed": 42
}

# --- Sweep Configuration () ---
config_sweep = {
    **config_base,
    "batch_size": 64,
    "block_size": 64,
    "max_iters": 500,
    "eval_interval": 20,
    "n_embed": 64,
    "n_head": 4,
    "n_layer": 3,
    "num_experts": 4,
}
# --- Full Run Configuration ---
config_full = {
    **config_base,
    "batch_size": 128,
    "block_size": 128,
    "max_iters": 5000,
    "eval_interval": 100,
    "n_embed": 128,
    "n_head": 8,
    "n_layer": 6,
    "num_experts": 8,
    "router_alpha": config_base["router_alpha"],
    "router_temperature": config_base["router_temperature"],
}

config_full["eval_iters"] = (config_full["max_iters"] // config_full["eval_interval"]) * config_full[
    "eval_iters_scale_factor"]

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

    # Split for SWEEP runs (using 90% of full data) - as defined in notebook
    n_split = int(0.9 * len(data))
    train_data = data[:n_split]  # This will be used by the sweep
    val_data = data[n_split:]  # This will be used by the sweep
    custom_print(f"Sweep Data Split: Train tokens: {len(train_data)}, Valid tokens: {len(val_data)}")

except Exception as e:
    custom_print(f"Error processing dataset file: {e}")
    exit()


# --- Model Definitions  ---

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
        assert temperature > 1e-9
        self.num_experts = num_experts
        self.alpha = alpha
        self.temperature = temperature
        self.n_iter = n_iter
        self.route_linear = nn.Linear(n_embed, num_experts)
        if not ENTMAX_AVAILABLE and self.alpha != 1.0:
            custom_print(f"Warning: entmax library not found, alpha={self.alpha} runs will use Softmax fallback.")

    def forward(self, x):
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
        else:
            router_output = F.softmax(scaled_logits, dim=-1)

        if router_output is None:
            custom_print("ERROR: router_output logic failed. Defaulting to Softmax.")
            router_output = F.softmax(scaled_logits, dim=-1)

        return router_output


class SparseMoE(nn.Module):
    """ Sparse Mixture of Experts layer """

    def __init__(self, n_embed, num_experts, dropout, router_alpha=1.5, router_temperature=1.0):
        super().__init__()
        self.router = AlphaEntmaxRouter(n_embed, num_experts, alpha=router_alpha, temperature=router_temperature)
        self.experts = nn.ModuleList([Expert(n_embed, dropout) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.register_buffer('latest_tpe', torch.zeros(num_experts, dtype=torch.float32))
        self.latest_imbalance: float = 1.0
        self.latest_concentration: float = 0.0
        self.latest_utilization: float = 0.0
        self.latest_tpe_cv: float = 0.0
        self.latest_avg_prob_cv: float = 0.0
        self.activation_threshold = 1e-9

    def calculate_metrics(self, gating_output_no_grad):
        """ Calculates routing metrics based on detached gating output """
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
            is_active = gating_output_no_grad > self.activation_threshold
            tpe = is_active.sum(dim=0).float()
            if self.latest_tpe.shape[0] == tpe.shape[0]:
                self.latest_tpe.copy_(tpe)
            else:
                custom_print(f"Warning: TPE size mismatch. Re-registering buffer.")
                self.register_buffer('latest_tpe', tpe.clone())

            if self.num_experts > 1:
                mean_tpe = tpe.mean()
                std_tpe = tpe.std(unbiased=False)
                self.latest_tpe_cv = (std_tpe / (mean_tpe + 1e-9)).item() if mean_tpe > 1e-9 else 0.0
            else:
                self.latest_tpe_cv = 0.0

            mean_tpe_val = tpe.mean().item()
            if mean_tpe_val > 0:
                self.latest_imbalance = tpe.max().item() / (mean_tpe_val + 1e-9)
            else:
                self.latest_imbalance = 1.0

            max_p_per_token, _ = gating_output_no_grad.max(dim=-1)
            self.latest_concentration = max_p_per_token.mean().item()

            num_active_experts = (tpe > 0).sum().item()
            self.latest_utilization = num_active_experts / self.num_experts if self.num_experts > 0 else 0.0

            if self.num_experts > 1:
                avg_prob = gating_output_no_grad.mean(dim=0)
                mean_avg_prob = avg_prob.mean()
                std_avg_prob = avg_prob.std(unbiased=False)
                self.latest_avg_prob_cv = (
                            std_avg_prob / (mean_avg_prob + 1e-9)).item() if mean_avg_prob > 1e-9 else 0.0
            else:
                self.latest_avg_prob_cv = 0.0

    def forward(self, x):
        batch_size, seq_len, n_embed = x.shape
        num_tokens = batch_size * seq_len
        x_reshaped = x.view(num_tokens, n_embed)
        gating_output = self.router(x_reshaped)
        self.calculate_metrics(gating_output.detach())
        final_output = torch.zeros_like(x_reshaped)
        for i, expert in enumerate(self.experts):
            token_indices = torch.nonzero(gating_output[:, i] > self.activation_threshold).squeeze(-1)
            if token_indices.numel() == 0: continue
            expert_input = x_reshaped[token_indices]
            active_gating_scores = gating_output[token_indices, i].unsqueeze(1)
            expert_output = expert(expert_input)
            weighted_output = expert_output * active_gating_scores
            final_output.index_add_(0, token_indices, weighted_output)
        final_output = final_output.view(batch_size, seq_len, n_embed)
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
        x = x + self.self_attn(self.ln1(x))
        x = x + self.sparse_moe(self.ln2(x))
        return x


class SparseMoELanguageModel(nn.Module):
    """ Language Model using SparseMoE Blocks """

    def __init__(self, vocab_size, n_embed, n_head, n_layer, num_experts, block_size, dropout, router_alpha=1.5,
                 router_temperature=1.0):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed=n_embed, n_head=n_head, num_experts=num_experts, block_size=block_size, dropout=dropout,
                  router_alpha=router_alpha, router_temperature=router_temperature)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=device)

        if T > self.block_size:
            custom_print(f"Warning: Input sequence length T={T} exceeds block_size={self.block_size}. Truncating.")
            pos = pos[-self.block_size:]
            tok_emb = tok_emb[:, -self.block_size:, :]
            T = self.block_size

        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B_logits, T_logits, C_logits = logits.shape
            if targets.shape[1] > T_logits:
                targets = targets[:, -T_logits:]

            logits_flat = logits.view(B_logits * T_logits, C_logits)
            targets_flat = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# --- Utility Functions ---

def get_batch(split, data_train, data_val, block_size, batch_size, device):
    """ Generate a small batch of data of inputs x and targets y """
    data_source = data_train if split == 'train' else data_val
    max_start_index = len(data_source) - block_size - 1
    if max_start_index < 0:
        custom_print(
            f"Warning: Dataset split length ({len(data_source)}) is smaller than block_size ({block_size}). Cannot generate batch.")
        return None, None

    ix = torch.randint(max_start_index + 1, (batch_size,))
    x = torch.stack([data_source[i: i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, config, data_train, data_val):
    """ Estimate loss and metrics on train/val splits """
    out = {'loss': {}}
    eval_iters = config["eval_iters"]
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    n_layer = config["n_layer"]
    num_experts = config["num_experts"]
    device = next(model.parameters()).device
    model.eval()

    val_metrics_accumulated = {
        f'L{layer_idx}': {'imbalance': 0.0, 'concentration': 0.0, 'utilization': 0.0,
                          'tpe_cv': 0.0, 'avg_prob_cv': 0.0, 'count': 0}
        for layer_idx in range(n_layer)
    }

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'val':
            for layer_idx in range(n_layer):
                layer_metrics = val_metrics_accumulated[f'L{layer_idx}']
                for key in layer_metrics:
                    layer_metrics[key] = 0.0 if key != 'count' else 0
        for k in range(eval_iters):
            xb, yb = get_batch(split, data_train, data_val, block_size, batch_size, device)
            if xb is None or yb is None:
                losses[k] = float('nan')
                continue

            logits, loss = model(xb, yb)
            losses[k] = loss.item() if loss is not None else float('nan')

            if split == 'val' and hasattr(model, 'blocks') and isinstance(model.blocks, nn.Sequential):
                for i, block in enumerate(model.blocks):
                    if hasattr(block, 'sparse_moe') and isinstance(block.sparse_moe, SparseMoE):
                        smoe_layer = block.sparse_moe
                        layer_key = f'L{i}'
                        layer_metrics = val_metrics_accumulated[layer_key]
                        if hasattr(smoe_layer, 'latest_imbalance'):
                            # Accumulate metrics only if they are valid numbers
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
                            layer_metrics['count'] += 1

        out['loss'][split] = np.nanmean(losses.numpy()) if not torch.all(torch.isnan(losses)) else float('nan')

    total_metrics_sum = {'imbalance': 0.0, 'concentration': 0.0, 'utilization': 0.0,
                         'tpe_cv': 0.0, 'avg_prob_cv': 0.0}
    total_layers_with_metrics = 0

    for i in range(n_layer):
        count = val_metrics_accumulated[f'L{i}']['count']
        if count > 0:
            total_layers_with_metrics += 1
            metrics_sum = val_metrics_accumulated[f'L{i}']
            for key in total_metrics_sum:
                metric_value = metrics_sum[key]
                if not np.isnan(metric_value):
                    avg_layer_metric = metric_value / count
                    total_metrics_sum[key] += avg_layer_metric

    overall_avg_metrics = {}
    if total_layers_with_metrics > 0:
        for key in total_metrics_sum:
            overall_avg_metrics[key] = total_metrics_sum[key] / total_layers_with_metrics
    else:
        for key in total_metrics_sum:
            overall_avg_metrics[key] = 0.0

    model.train()
    return out['loss'], overall_avg_metrics


# ============================================================
# --- SECTION 1: Hyperparameter Sweep ---
# ============================================================

custom_print(f"\n--- Running Hyperparameter Sweep ---")

alphas = [1.0, 1.5, 2.0, 2.5]
temperatures = [0.5, 1.0, 10.0]
num_seeds = 5
base_seed = config_base["seed"]
config = config_full

custom_print(f"Sweep Config: {config}")
custom_print(f"Sweeping Alphas: {alphas}")
custom_print(f"Sweeping Temperatures: {temperatures}")
custom_print(f"Running {num_seeds} seeds per combination.")

sweep_train_data = train_data
sweep_val_data = val_data

dtypes = {'model_key': str, 'alpha': float, 'temperature': float, 'seed': int, 'iter': int,
          'loss_train': float, 'loss_val': float,
          'Expert Utilization': float, 'Load Imbalance Ratio': float, 'Routing Concentration': float,
          'Expert Load Variation': float, 'Routing Probability CV': float}
df_sweep_results = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})

sweep_start_time = time.time()
import glob

# --- Sweep Loops (with Multi-Seed) ---
for alpha in alphas:
    for temperature in temperatures:
        for seed_run_index in range(num_seeds):  # Iterate through seeds
            current_seed = base_seed + seed_run_index
            run_key = f"alpha_{alpha}_temp_{temperature}_seed_{current_seed}"

            if len(glob.glob(f"./*{run_key}*.csv")) > 0:
                custom_print(f"skipping processing : {run_key}")
                continue
            else:
                file_name = f"{run_key}_Running_{str(sweep_start_time)}.csv"  # delete this manually after
                df_sweep_results.to_csv(file_name)  # just to be fail safe and avoid re-experimentation

            custom_print(f"\n----- Starting Sweep Run: {run_key} -----")

            custom_print(f"Setting seed: {current_seed}")
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)  # Seed numpy as well

            config['router_alpha'] = alpha
            config['router_temperature'] = temperature

            # --- Re-initialize Model and Optimizer for each seed ---
            model = SparseMoELanguageModel(
                vocab_size=full_vocab_size,
                n_embed=config["n_embed"], n_head=config["n_head"], n_layer=config["n_layer"],
                num_experts=config["num_experts"], block_size=config["block_size"], dropout=config["dropout"],
                router_alpha=config["router_alpha"], router_temperature=config["router_temperature"]
            ).to(device)
            custom_print(f"Sweep Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

            run_start_time = time.time()
            # --- Training Loop ---
            for iter_num in range(config["max_iters"]):
                if iter_num % config["eval_interval"] == 0 or iter_num == config["max_iters"] - 1:
                    losses, avg_metrics_val = estimate_loss(model, config, sweep_train_data, sweep_val_data)
                    loss_train_val = losses.get('train', float('nan'))
                    loss_val_val = losses.get('val', float('nan'))

                    new_row_data = {
                        'model_key': f"alpha_{alpha}_temp_{temperature}",
                        'alpha': alpha, 'temperature': temperature, 'seed': current_seed, 'iter': iter_num,
                        'loss_train': loss_train_val, 'loss_val': loss_val_val,
                        'Expert Utilization': avg_metrics_val.get('utilization', 0.0),
                        'Load Imbalance Ratio': avg_metrics_val.get('imbalance', 0.0),
                        'Routing Concentration': avg_metrics_val.get('concentration', 0.0),
                        'Expert Load Variation': avg_metrics_val.get('tpe_cv', 0.0),
                        'Routing Probability CV': avg_metrics_val.get('avg_prob_cv', 0.0)
                    }
                    new_row = pd.DataFrame([new_row_data])
                    # --- Append results safely (Handles FutureWarning) ---
                    if not new_row.empty and not new_row.isnull().all().all():
                        df_sweep_results = pd.concat([df_sweep_results, new_row], ignore_index=True)
                    # --- ---

                    # Use exact custom_print format requested
                    custom_print(
                        f"Sweep Run: {run_key} | Iter: {iter_num:<4}/{config['max_iters']:<4} || Est Train Loss: {loss_train_val:.4f} | Val Loss: {loss_val_val:.4f}")
                    custom_print(
                        f"Metrics (Avg) || Expert Utilization: {avg_metrics_val.get('utilization', 0.0):<6.2f} | Load Imbalance Ratio: {avg_metrics_val.get('imbalance', 0.0):<6.2f} | Routing Concentration: {avg_metrics_val.get('concentration', 0.0):<6.2f} | Expert Load Variation: {avg_metrics_val.get('tpe_cv', 0.0):<6.2f} | Routing Probability CV: {avg_metrics_val.get('avg_prob_cv', 0.0):<6.2f}")

                xb, yb = get_batch('train', sweep_train_data, sweep_val_data, config["block_size"],
                                   config["batch_size"], device)
                if xb is None or yb is None: continue  # Skip if batch failed
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                file_name = f"all_output_results_{str(sweep_start_time)}.csv"
                df_sweep_results.to_csv(file_name)
            file_name = f"{run_key}_Completed_{str(sweep_start_time)}.csv"  # delete this manually after
            df_sweep_results.to_csv(file_name)  # just to be fail safe and avoid re-experimentation
            run_end_time = time.time()
            custom_print(f"----- Sweep Run {run_key} completed in {run_end_time - run_start_time:.2f} seconds -----")

sweep_end_time = time.time()
custom_print(f"\n--- Full Sweep (all seeds) completed in {sweep_end_time - sweep_start_time:.2f} seconds ---")

sweep_results_file = 'csv_out/sweep_results_all_seeds.csv'
custom_print(f"\nSaving detailed sweep results to {sweep_results_file}")
df_sweep_results.to_csv(sweep_results_file, index=False)

custom_print("\n--- Aggregated Sweep Results (Averaged Across Seeds) ---")
final_iter_sweep = config_sweep["max_iters"] - 1
df_final_iter_sweep = df_sweep_results[df_sweep_results['iter'] == final_iter_sweep].copy()

metric_cols_display = ['loss_val', 'Expert Utilization', 'Load Imbalance Ratio',
                       'Routing Concentration', 'Expert Load Variation', 'Routing Probability CV']

best_sweep_params = None
if not df_final_iter_sweep.empty:
    grouped_sweep_results = df_final_iter_sweep.groupby(['alpha', 'temperature'])
    aggregated_sweep_stats = grouped_sweep_results[metric_cols_display].agg(['mean', 'std'])
    aggregated_sweep_stats.columns = ['_'.join(col).strip() for col in aggregated_sweep_stats.columns.values]
    aggregated_sweep_stats = aggregated_sweep_stats.sort_values('loss_val_mean')

    custom_print(aggregated_sweep_stats)
    agg_sweep_results_file = 'sweep_results_aggregated.csv'
    custom_print(f"\nSaving aggregated sweep results to {agg_sweep_results_file}")
    aggregated_sweep_stats.to_csv(agg_sweep_results_file)

    best_params_idx = aggregated_sweep_stats['loss_val_mean'].idxmin()
    best_sweep_params = {'alpha': best_params_idx[0], 'temperature': best_params_idx[1]}
    custom_print(
        f"\nBest parameters found from sweep: Alpha={best_sweep_params['alpha']}, Temperature={best_sweep_params['temperature']}")
else:
    custom_print("No final iteration results found from sweep to aggregate.")
