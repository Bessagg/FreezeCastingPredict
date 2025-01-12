from data_parser import DataParser
from helpers import utils
import warnings
import glob
import os
from scipy.stats import kstest
import matplotlib
matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr, chi2_contingency
from scipy.stats import chi2_contingency, kruskal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 600)
im_path = 'images/model_analysis'


def plot_error_distribution_by_group(df, error: pd.Series, group_column, title=""):
    # Create a save directory for images
    save_dir = f'images/results/{title}_error_by_group_{group_column}.png'

    # Create a figure with 3 subplots (1 column, 3 rows)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 18), sharex=True)

    # Define group names and their corresponding colors
    if group_column == "name_fluid1":
        groups = ['water', 'TBA', 'camphene']
    else:
        groups = ['Ceramic', 'Metal', 'Polymer']

    colors = ['blue', 'red', 'green']

    # Define custom bin edges with a step of 0.01
    bins = np.arange(-0.4, 0.401, 0.01)  # Bins from -0.4 to 0.4 with step size of 0.01

    # Loop through the groups to plot error distribution for each group
    for i, group in enumerate(groups):
        # Filter the dataframe for the current group
        group_df = df[df[group_column] == group]

        # Get the error values for the current group
        group_error = error[group_df.index]  # Use the error values for the current group

        # Plot the histogram on the corresponding subplot
        sns.histplot(group_error, bins=bins, color=colors[i], ax=axes[i], kde=False)

        # Set labels and title for the subplot
        axes[i].set_xlabel("Error", fontsize=32)
        axes[i].set_ylabel("Sample Count", fontsize=32)
        axes[i].tick_params(labelsize=22)
        axes[i].set_title(f"Error Distribution: {group}", fontsize=32)
        axes[i].set_xlim(-0.4, 0.4)  # Limit x-axis to match the range

        # Calculate mean and standard deviation for the current group
        mean_error = np.mean(group_error)
        std_error = np.std(group_error)

        # Add mean and std dev as text labels on the plot
        axes[i].text(0.1, 0.85, f'Mean: {mean_error:.2f}', ha='left', va='center', transform=axes[i].transAxes,
                     fontsize=32)
        axes[i].text(0.1, 0.75, f'±1 Std Dev: {std_error:.2f}', ha='left', va='center', transform=axes[i].transAxes,
                     fontsize=32)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)  # Increase vertical spacing
    plt.tick_params(labelsize=28)  # Set font size for tick labels
    # Tight layout to avoid overlap of titles and labels
    plt.tight_layout()

    # Show and save the plot
    plt.show()

    # Save the plot to the specified directory
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir, bbox_inches='tight')


def plot_prediction_performance_by_group(true_values, predicted_values, group_column):
    save_dir = f'images/results/perf_by_group.png'
    # Create a dataframe for easier plotting with seaborn
    df = pd.DataFrame({
        'True': true_values,
        'Predicted': predicted_values,
        'Group': group_column
    })

    # Calculate the error (difference between true and predicted)
    error = np.abs(true_values - predicted_values)

    # Create a custom colormap directly from green to yellow to red
    colors = ['darkgreen', 'green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('green_yellow_red', colors, N=n_bins)

    # Create the plot
    plt.figure(figsize=(18, 18))

    # Create a grid of points
    xi = np.linspace(-0.1, 1.1, 50)
    yi = np.linspace(-0.1, 1.1, 50)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate error values across the grid
    Z = griddata((df['True'], df['Predicted']), error, (X, Y), method='cubic')

    # Create a normalization to match the error range
    norm = Normalize(vmin=0, vmax=np.max(error))

    # Create a heatmap for the background
    im = plt.imshow(Z, extent=[-0.1, 1.1, -0.1, 1.1],
                    origin='lower',
                    cmap=cmap,
                    norm=norm,
                    alpha=0.3,  # Transparency of the background
                    aspect='auto')

    # Define distinct colors for each group
    group_colors = ['blue', 'red', 'green']

    # Define distinct markers for each group
    group_markers = ['o', 's', '^']  # circle, square, triangle

    # Get the most frequent groups
    top_groups = df['Group'].value_counts().nlargest(3).index.tolist()

    # Plot the scatterplot with group-specific colors and markers
    for i, group in enumerate(top_groups):
        group_df = df[df['Group'] == group]

        plt.scatter(group_df['True'], group_df['Predicted'],
                    color=group_colors[i],  # Color by group
                    edgecolor='black',
                    marker=group_markers[i],  # Use different marker for each group
                    alpha=0.7,
                    label=group,
                    s=100)  # Marker size

    # Add line for perfect prediction (y = x)
    plt.plot([-0.1, 1.1], [-0.1, 1.1],
             color='black',
             linestyle='--',
             label='Perfect Prediction')

    # Add labels and title
    plt.xlabel('True Values', fontsize=32)
    plt.ylabel('Predicted Values', fontsize=32)
    plt.title('True vs Predicted Values by Group', fontsize=32)

    # Set x and y axis limits
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    # Set custom tick marks
    ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tick_params(labelsize=28)  # Set font size for tick labels

    # Add color bar to indicate error levels
    plt.colorbar(im, label='Prediction Error')

    # Add legend and adjust its properties
    plt.legend(title='Groups', title_fontsize=32, fontsize=32, loc='best')
    # Tight layout to prevent cutting off labels
    plt.tight_layout()

    # Display the plot
    plt.show()
    plt.savefig(save_dir, bbox_inches='tight')


def plot_prediction_performance(true_y, prediction: pd.Series, error: pd.Series, title=""):
    save_path = f'images/results/{title}_perf.png'
    palette = sns.color_palette("RdYlGn_r", as_cmap=True)  # 'RdYlGn_r' reverses the palette (green to red)
 # Red-focused colormap
    # Model performance plot
    plt.figure(figsize=(14, 10))
    plt.clf()
    ax = sns.scatterplot(x=true_y, y=prediction, hue=error,
                         palette=palette)
    norm = plt.Normalize(0, 0.4)  # set min and max for color_bar
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    ax.set_xlabel("True Porosity", fontsize=20)
    ax.set_ylabel(f"Predicted Porosity", fontsize=20)
    ax.get_legend().remove()
    ax.figure.colorbar(sm)
    ax.set(ylim=(-0.1, 1.11))
    ax.set(xlim=(-0.1, 1.1))
    ax.tick_params(labelsize=28)
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.show()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')


def plot_error_distribution(df, error: pd.Series, title=""):
    save_dir = f'images/results/{title}_error.png'
    plt.figure(figsize=(18, 12))
    bx = sns.histplot(data=df, x=error,  # hue="Group",
                      bins=20,
                      palette=sns.color_palette("hls", 3))
    bx.set_xlabel("Error", fontsize=20)
    bx.set_ylabel(f"Sample count", fontsize=20)
    bx.tick_params(labelsize=28)
    bx.set(xlim=(-0.4, 0.4))
    x_axis = [round(num, 2) for num in np.linspace(-0.4, 0.4, 7)]
    plt.show()
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir, bbox_inches='tight')


def filter_topn_categories(df, column_name, n=3):
    top3_group = df[column_name].value_counts().iloc[:n].index.to_list()
    return df[(df[column_name].isin(top3_group))]


def ks_test_group(group_df, column, distribution='norm'):
    # Perform the one-sample K-S test
    statistic, p_value = kstest(group_df[column], distribution)
    return pd.Series({'ks_statistic': statistic, 'p_value': p_value})


"""General Setup"""
print("Parsing data...")
parser = DataParser()
target = parser.target

"""Fetch dataset"""
print("Loading Data...")
selected_feats_name = "reduced_feats"
df_train = pd.read_csv(f"data/{selected_feats_name}/train.csv")
df_test = pd.read_csv(f"data/{selected_feats_name}/test.csv")

"""Get model paths from selected_models"""
selected_models_dir = f'selected_models/{selected_feats_name}'
pickle_pipeline_paths = glob.glob(os.path.join(selected_models_dir, '**', 'model', '*.pickle'), recursive=True)

"""Load models"""
selected_pipes = []
for pipe_path in pickle_pipeline_paths:
    pipe = parser.load_pipeline(pipe_path)
    selected_pipes.append(pipe)

"""Evaluate models"""
results_models_dict_list = []
results_importances = []
results_groups = []
for pipe in selected_pipes:
    print(pipe.name, pipe.feats_name)
    pipe_ggparent_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pipe_path))))
    train_preds, test_preds = pipe.predict(df_train), pipe.predict(df_test)
    df_test[f'{pipe.name}_prediction'] = test_preds
    mae_test = np.abs(test_preds - df_test[target])
    df_test[f'{pipe.name}_mae'] = mae_test
    print("Train results")
    r2_train, mae_train, mse_train, mape_train = utils.get_regression_metrics(train_preds, df_train[target])
    print("Test results")
    r2_test, mae_test, mse_test, mape_test = utils.get_regression_metrics(test_preds, df_test[target])
    print("Test nulls results")
    columns_not_null = ['name_part1', 'name_fluid1', 'vf_total', 'material_group', target]
    df_test_nulls = df_test.copy()
    df_test_filtered = df_test.copy()
    print("Other feats are null count", len(df_test_filtered))
    df_test_filtered = df_test_filtered[
        df_test_filtered[pipe.selected_feats].notna().any(axis=1)  # Keep rows where any of the features are non-null
        & df_test_filtered[columns_not_null].notna().all(axis=1)
        # Keep rows where all columns of interest are non-null
        ]

    test_preds_nulls = pipe.predict(df_test_filtered)
    r2_test_nulls, mae_test_nulls, mse_test_nulls, mape_test_nulls = utils.get_regression_metrics(test_preds_nulls, df_test[target])

    # Add model results
    result_model_dict = {'name': pipe.name, 'r2_train': r2_train, 'mae_train': mae_train, 'mse_train': mse_train,
                         'r2_test': r2_test, 'mae_test': mae_test, 'mse_test': mse_test}
    results_models_dict_list.append(result_model_dict)
    preds_train, preds_test = pipe.predict(df_train), pipe.predict(df_test)
    results_train = {'name': pipe.name, 'feats': pipe.feats_name, 'prediction_train': preds_train,
                     'prediction_test': preds_test,
                     'train_true': df_train[target], 'test_true': df_test[target],
                     'model_type': '-', }

    # Add importances if available
    estimator = pipe[-1]
    if hasattr(estimator, 'feature_importances_'):
        feature_importances = estimator.feature_importances_
        # normalize
        feature_importances = feature_importances / feature_importances.sum()
        feature_names = pipe[0].get_feature_names_out()
        feature_names = np.array([name.replace('remainder__', '').replace('cat__', '') for name in feature_names])

        for i, importance in enumerate(feature_importances):
            results_importances.append({
                'model': pipe.name,
                'feature_name': feature_names[i],
                'feature_importance': importance
            })

    # Calculate metrics grouped by the specified feature
    groupby_features = ['material_group', 'name_part1', 'name_fluid1']
    for groupby_feature in groupby_features:
        grouped = df_test.groupby(groupby_feature)
        grouped_train = df_train.groupby(groupby_feature)
        category_counts = grouped_train[groupby_feature].value_counts().sort_values(ascending=False)
        # Determine top 5 most frequent categories
        top_categories = category_counts[0:5]

        for category, category_indices in grouped.groups.items():
            # just for tagging top5 if category in most frequent and not filtering
            tag_top5 = category in top_categories
            if not tag_top5:
                continue
            group_df = df_test.loc[category_indices]
            group_y_true = df_test[target][category_indices]
            group_y_pred = preds_test[category_indices]
            grouped_r2, grouped_mae, grouped_mse, grouped_mape = (
                utils.get_regression_metrics(group_y_pred, group_y_true, opt_print=False))
            group_variance = round(np.var(group_y_pred), 3)
            group_std = round(np.std(group_y_true), 2)  # true porosity std
            # append results
            results_groups.append({
                'model': pipe.name,
                'groupby_feature': groupby_feature,
                'category': category,
                'train samples': df_train[groupby_feature].value_counts()[category],
                'test_samples': df_test[groupby_feature].value_counts()[category],
                'r2': grouped_r2,
                'mae': grouped_mae,
                'mse': grouped_mse,

                'variance': group_variance,
                'std': group_std,
                'mape': grouped_mape,
                'top5': tag_top5,
            })

    print('\n')

"""Plots"""
selected_model = 'catb'
df_results = pd.DataFrame(results_groups)
df_imps = pd.DataFrame(results_importances)
df_groups = pd.DataFrame(results_groups)

# Filter metrics of top5 most frequent categories
df_groups_filtered = df_groups.query("model == 'catb' and top5 == True").sort_values(
    by=['groupby_feature', 'train samples'], ascending=False)
print("Selected model results by group:")
print(df_groups_filtered)

print("Relationship between STD")
metric = 'train samples'
df_groups_filtered['variance'] = df_groups_filtered[metric].astype(float)
df_groups_filtered['std'] = df_groups_filtered['std'].astype(float)
corr = df_groups_filtered['variance'].corr(df_groups_filtered['std'])
plt.scatter(df_groups_filtered['r2'].astype(float), df_groups_filtered['std'].astype(float))
plt.xlabel(r'$R^2$ value')
plt.ylabel("STD")
X = df_groups_filtered[metric]
y = df_groups_filtered['r2']
plt.clf()
plt.scatter(X, y, color='blue', label='Grouped data')
plt.xlabel('Standard Deviation of true porosity')
plt.ylabel(r'$R^2$ Value')
plt.title('Linear Regression between STD and $R^2$')
plt.legend()
plt.show()
plt.savefig(f"{im_path}/r2_vs_std.png")

## Performance Plots
selected_model_name = "catb"
prediction = df_test[f"{selected_model_name}_prediction"]
true_y = df_test[parser.target]
error = df_test[f"{selected_model_name}_prediction"] - df_test[parser.target]
plot_prediction_performance(df_test[parser.target], df_test[f"{selected_model_name}_prediction"], df_test[f"{selected_model_name}_mae"], title=selected_model_name)
plot_error_distribution(df_test, error)
plot_prediction_performance_by_group(df_test[parser.target], df_test[f"{selected_model_name}_prediction"], df_test['name_fluid1'])
plot_prediction_performance_by_group(df_test[parser.target], df_test[f"{selected_model_name}_prediction"], df_test['material_group'])

plot_error_distribution_by_group(df_test, error, 'name_fluid1')
plot_error_distribution_by_group(df_test, error, 'material_group')
plt.close('all')
