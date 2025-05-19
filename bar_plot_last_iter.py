import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Modern color palette
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']

# File mapping for each alpha
alpha_files = {
    '1.0': './results_csv/alpha_1_0.csv',
    '1.5': './results_csv/alpha_1_5.csv',
    '2.0': './results_csv/alpha_2_0.csv',
    '2.5': './results_csv/alpha_2_5.csv'
}


results = []

for alpha, file_path in alpha_files.items():
    df = pd.read_csv(file_path)

    # Find the last iteration value for each (temperature, seed)
    last_iter = df['iter'].max()
    last_df = df[df['iter'] == last_iter]

    # Compute mean and std of loss_val at last iteration per temperature
    summary = last_df.groupby(['temperature'])['loss_val'].agg(['mean', 'std']).reset_index()
    summary['alpha'] = alpha

    results.append(summary)

# Combine all summaries
all_data = pd.concat(results)

# Convert to categorical for plotting
all_data['temperature'] = all_data['temperature'].astype(str)
all_data['alpha'] = all_data['alpha'].astype(str)

# Pivot for grouped bar plot
pivot_mean = all_data.pivot(index='temperature', columns='alpha', values='mean')
pivot_std = all_data.pivot(index='temperature', columns='alpha', values='std')

# Sort temperatures and alphas
temperatures = sorted(pivot_mean.index.tolist(), key=float)
alphas = sorted(pivot_mean.columns.tolist(), key=float)

# Plot settings
x = np.arange(len(temperatures))
width = 0.18

plt.figure(figsize=(10, 6))

# Plot each alpha's bar at final iter
for i, alpha in enumerate(alphas):
    means = pivot_mean.loc[temperatures, alpha]
    stds = pivot_std.loc[temperatures, alpha]
    plt.bar(x + i * width, means, width, yerr=stds, capsize=4, label=f'alpha={alpha}',
            color=colors[i % len(colors)])

# X-axis labels
plt.xticks(x + width * (len(alphas) - 1) / 2, temperatures)
plt.xlabel('Temperature')
plt.ylabel('Validation Loss (Final Iteration)')
plt.title('Validation Loss at Final Iteration by Temperature and Alpha (Mean ± Std)')
plt.legend(title='Alpha')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Zoom y-axis to highlight differences
min_val = all_data['mean'].min()
max_val = all_data['mean'].max()
plt.ylim(bottom=min_val - 0.01, top=max_val + 0.01)

plt.tight_layout()
plt.savefig("./output/last_iter_val_loss_bar.png", dpi=600)
plt.show()
