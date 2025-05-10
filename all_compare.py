import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv("results_csv/alpha_1_0.csv")
df2 = pd.read_csv("results_csv/alpha_1_5.csv")
df3 = pd.read_csv("results_csv/alpha_2_0.csv")
df4 = pd.read_csv("results_csv/alpha_2_5.csv")


# Combine all into one DataFrame
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Define metrics and unique temperatures
metric_cols = ['loss_train', 'loss_val', 'Expert Utilization', 'Load Imbalance Ratio',
               'Routing Concentration', 'Expert Load Variation', 'Routing Probability CV']
temps = sorted(df['temperature'].unique())

# Compute mean and std across seeds
group_mean = df.groupby(['temperature', 'alpha', 'iter'])[metric_cols].mean().reset_index()
group_std = df.groupby(['temperature', 'alpha', 'iter'])[metric_cols].std().reset_index()

# New layout: rows = metrics, cols = temperatures
n_rows = len(metric_cols)
n_cols = len(temps)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex='col')
axes = axes.reshape(n_rows, n_cols)

# Plot
for i, metric in enumerate(metric_cols):
    for j, temp in enumerate(temps):
        ax = axes[i, j]
        for alpha in sorted(df['alpha'].unique()):
            mean_sub = group_mean[(group_mean['temperature'] == temp) & (group_mean['alpha'] == alpha)].sort_values('iter')
            std_sub = group_std[(group_std['temperature'] == temp) & (group_std['alpha'] == alpha)].sort_values('iter')

            ax.errorbar(mean_sub['iter'], mean_sub[metric], yerr=std_sub[metric],
                        label=f'α={alpha}', capsize=3)

        if i == 0:
            ax.set_title(f"Temp {temp}")
        if j == 0:
            ax.set_ylabel(metric)
        ax.grid(True)
        if i == 0 and j == 0:
            ax.legend(fontsize='small')

fig.suptitle("Metric Comparison for Different α (Columns = Temp, Rows = Metric)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('./output/all_compared.png', dpi=300)
plt.show()
