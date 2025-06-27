import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# from train pipeline bin  print
data = [
    # (Model, Freq Bin, Train, Test, R2, MAE)
    ("CatBoost (one-hot + impute)", "0", 0, 4, -11.10, 0.242),
    ("CatBoost (one-hot)", "0", 0, 4, -11.93, 0.252),
    ("XGBoost (one-hot)", "0", 0, 4, -12.68, 0.254),
    ("CatBoost (native)", "0", 0, 4, -12.83, 0.260),
    ("Random Forest", "0", 0, 4, -14.71, 0.277),
    ("XGBoost (one-hot + impute)", "0", 0, 4, -46.21, 0.480),

    ("CatBoost (native)", "≤5", 79, 26, 0.83, 0.055),
    ("XGBoost (one-hot + impute)", "≤5", 79, 26, 0.80, 0.061),
    ("CatBoost (one-hot)", "≤5", 79, 26, 0.84, 0.053),
    ("XGBoost (one-hot)", "≤5", 79, 26, 0.76, 0.074),
    ("CatBoost (one-hot + impute)", "≤5", 79, 26, 0.83, 0.056),
    ("Random Forest", "≤5", 79, 26, 0.81, 0.061),

    ("CatBoost (native)", "≤10", 73, 32, 0.49, 0.084),
    ("Random Forest", "≤10", 73, 32, 0.63, 0.069),
    ("CatBoost (one-hot)", "≤10", 73, 32, 0.65, 0.069),
    ("XGBoost (one-hot)", "≤10", 73, 32, 0.71, 0.069),
    ("XGBoost (one-hot + impute)", "≤10", 73, 32, 0.76, 0.057),
    ("CatBoost (one-hot + impute)", "≤10", 73, 32, 0.65, 0.071),

    ("CatBoost (native)", "≤50", 333, 67, 0.83, 0.067),
    ("XGBoost (one-hot + impute)", "≤50", 333, 67, 0.88, 0.051),
    ("CatBoost (one-hot)", "≤50", 333, 67, 0.89, 0.051),
    ("CatBoost (one-hot + impute)", "≤50", 333, 67, 0.88, 0.052),
    ("XGBoost (one-hot)", "≤50", 333, 67, 0.87, 0.057),
    ("Random Forest", "≤50", 333, 67, 0.86, 0.058),

    ("CatBoost (one-hot)", "≤200", 359, 77, 0.85, 0.051),
    ("XGBoost (one-hot + impute)", "≤200", 359, 77, 0.82, 0.056),
    ("Random Forest", "≤200", 359, 77, 0.82, 0.054),
    ("CatBoost (one-hot + impute)", "≤200", 359, 77, 0.85, 0.051),
    ("CatBoost (native)", "≤200", 359, 77, 0.78, 0.063),
    ("XGBoost (one-hot)", "≤200", 359, 77, 0.84, 0.052),

    ("CatBoost (one-hot)", "200+", 643, 166, 0.80, 0.059),
    ("Random Forest", "200+", 643, 166, 0.77, 0.063),
    ("XGBoost (one-hot + impute)", "200+", 643, 166, 0.77, 0.063),
    ("XGBoost (one-hot)", "200+", 643, 166, 0.76, 0.066),
    ("CatBoost (native)", "200+", 643, 166, 0.74, 0.067),
    ("CatBoost (one-hot + impute)", "200+", 643, 166, 0.79, 0.061),
]

df = pd.DataFrame(data, columns=["Model", "FreqBin", "Train", "Test", "R2", "MAE"])

def plot_freq_bin_model_perf(df, metric='R2', output_path=None, fontsize=20, dpi=300):
    """
    Plot model performance (R² or MAE) per frequency bin with test sample % on secondary axis.
    Each model is a bar per bin.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.ticker import FuncFormatter

    assert metric in ['R2', 'MAE'], "metric must be 'R2' or 'MAE'"

    bins_order = ["0", "≤5", "≤10", "≤50", "≤200", "200+"]
    df = df.copy()
    df["FreqBin"] = pd.Categorical(df["FreqBin"], categories=bins_order, ordered=True)

    # Pivot for bar plot: index=FreqBin, columns=Model, values=metric
    metric_pivot = df.pivot_table(index="FreqBin", columns="Model", values=metric)
    metric_pivot = metric_pivot.loc[bins_order]
    metric_pivot = metric_pivot.clip(lower=0)
    # % Test Samples per bin for line plot
    test_counts = df.groupby("FreqBin")["Test"].sum()
    pct_test = (test_counts / test_counts.sum() * 100).loc[bins_order]

    # Plot
    fig, ax1 = plt.subplots(figsize=(6.8*2, 4))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')


    from matplotlib.colors import LinearSegmentedColormap

    n_bins = len(metric_pivot.columns)
    cmap = LinearSegmentedColormap.from_list('green_blue', ["#00427d", "#ffffff", "#00652e"], N=n_bins)
    colors =  ["#00427d", "#00652e", "#E84855", "#f9dc5c",  "#461220", "#A37C40", "#BFCDE0", "#E3D26F"]
    colors =  [ "#00652e", "#A1C084", "#00427d", "#7F7979", "#E05263", "#DC9E82"]



    metric_pivot.plot(kind='bar', ax=ax1, color=colors, width=0.8, zorder=2)

    # Remove grid
   #ax1.grid(False)
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7, zorder=1)



    ax1.set_ylabel(f"Test $R^2$", fontsize=fontsize)
    # ax1.set_ylabel(f"Test {metric}", fontsize=fontsize)
    ax1.set_xlabel("Number of material's samples in the training set", fontsize=fontsize)
    ax1.set_xticklabels(bins_order, rotation=0, fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.legend(title="Model", fontsize=fontsize-2, title_fontsize=fontsize-2, bbox_to_anchor=(1.01, 1))

    # Secondary axis: % Test Samples (black line)
    ax2 = ax1.twinx()
    ax2.plot(bins_order, pct_test.values, color='black', marker='o', linestyle='-', linewidth=2, label='% of Total Test Samples', zorder=3)
    ax2.set_ylabel('% of Total Test Samples', fontsize=fontsize-1, color='black')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))

    # Remove y-axis tick labels on secondary axis
    ax2.tick_params(axis='y', labelleft=False, labelright=False, left=False, right=False,  pad=15)

    ax2.set_ylim(0, pct_test.max() * 1.15)

    # Add text labels inside plot for % Test Samples
    for x, y in zip(range(len(bins_order)), pct_test.values):
        ax2.text(x + 0.05, y + pct_test.max() * 0.03, f"{int(y)}%", color='black',
                 fontsize=fontsize - 4, ha='left', va='bottom')
        # Remove top and right spines
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # Scale axes
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, pct_test.max() * 1.15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    line2, = ax2.get_lines()
    ax1.legend(handles=lines1 + [line2], labels=labels1 + ['% of Total Test Samples'], loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=fontsize-2)

    # Move legend more right
    ax1.legend(handles=lines1 + [line2], labels=labels1 + ['% of Total Test Samples'],
               loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=fontsize - 2)


    plt.tight_layout()

    # Add margin on the right for legend
    plt.subplots_adjust(right=0.5)  # leave 25% for legend space

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
plot_freq_bin_model_perf(df, metric='R2', output_path="images/data_analysis" + "/material_frequency_bin.png")  # or 'MAE'