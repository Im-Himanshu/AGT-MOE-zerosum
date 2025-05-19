import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File mapping for each alpha
alpha_files = {
    '1.0': './results_csv/alpha_1_0.csv',
    '1.5': './results_csv/alpha_1_5.csv',
    '2.0': './results_csv/alpha_2_0.csv',
    '2.5': './results_csv/alpha_2_5.csv'

}

# Use a modern color palette (Tableau 10 or Seaborn's deep)
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # Modern & colorblind-friendly


results = []

# Process each CSV
for alpha, file_path in alpha_files.items():
    df = pd.read_csv(file_path)

    # Compute min val loss per (temperature, seed)
    min_loss_per_seed = df.groupby(['temperature', 'seed'])['loss_val'].min().reset_index()

    # Compute mean and std of min loss per temperature
    summary = min_loss_per_seed.groupby('temperature')['loss_val'].agg(['mean', 'std']).reset_index()
    summary['alpha'] = alpha

    results.append(summary)

# Combine all summaries
all_data = pd.concat(results)

# Ensure categorical types for plotting
all_data['temperature'] = all_data['temperature'].astype(str)
all_data['alpha'] = all_data['alpha'].astype(str)

# Pivot data for grouped bar plot
pivot_mean = all_data.pivot(index='temperature', columns='alpha', values='mean')
pivot_std = all_data.pivot(index='temperature', columns='alpha', values='std')

# Sort temperatures and alphas (optional)
temperatures = sorted(pivot_mean.index.tolist(), key=float)
alphas = sorted(pivot_mean.columns.tolist(), key=float)

# Plot configuration
x = np.arange(len(temperatures))  # positions of temperature groups
width = 0.18  # width of each bar

plt.figure(figsize=(10, 6))

# Plot each alpha's bar with error bar using modern colors
for i, alpha in enumerate(alphas):
    means = pivot_mean.loc[temperatures, alpha]
    stds = pivot_std.loc[temperatures, alpha]
    plt.bar(x + i * width, means, width, yerr=stds, capsize=4, label=f'alpha={alpha}',
            color=colors[i % len(colors)])

# Adjust x-axis
plt.xticks(x + width * (len(alphas) - 1) / 2, temperatures)
plt.xlabel('Temperature')
plt.ylabel('Minimum Validation Loss')
plt.title('Min Validation Loss by Temperature and Alpha (Mean ± Std)')
plt.legend(title='Alpha')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Zoom in y-axis to highlight differences
min_val = all_data['mean'].min()
max_val = all_data['mean'].max()
plt.ylim(bottom=min_val - 0.01, top=max_val + 0.01)

plt.tight_layout()
plt.savefig("./output/last_iter_val_loss_bar.png", dpi=300)
plt.show()
