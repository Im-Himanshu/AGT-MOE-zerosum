import pandas as pd
import matplotlib.pyplot as plt

alphas = ['1_0', '1_5', '2_0', '2_5']
for alpha in alphas:
    # Load data
    df = pd.read_csv(f'./alpha_{alpha}.csv')

    # Metrics to plot
    metric_cols = ['loss_train', 'loss_val', 'Expert Utilization', 'Load Imbalance Ratio',
                   'Routing Concentration', 'Expert Load Variation', 'Routing Probability CV']

    # Group by temperature and iter
    mean_df = df.groupby(['temperature', 'iter'])[metric_cols].mean().reset_index()
    std_df = df.groupby(['temperature', 'iter'])[metric_cols].std().reset_index()

    # Grid setup
    n_cols = 3
    n_rows = -(-len(metric_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    # Plot with error bars
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        for temp in sorted(df['temperature'].unique()):
            mean_temp_df = mean_df[mean_df['temperature'] == temp].sort_values('iter')
            std_temp_df = std_df[std_df['temperature'] == temp].sort_values('iter')

            ax.errorbar(mean_temp_df['iter'], mean_temp_df[metric],
                        yerr=std_temp_df[metric],
                        label=f'temp={temp}', capsize=3)

        ax.set_title(metric)
        ax.set_xlabel("iter")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(fontsize='small')

    # Hide unused subplots
    for j in range(len(metric_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Metrics vs Iter (Mean ± Std over Seeds) for Alpha : {alpha}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
