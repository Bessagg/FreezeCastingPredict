from data_parser import DataParser
from helpers import utils
import warnings
import glob
import os
from scipy.stats import kstest
from sklearn.metrics import r2_score
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
palette = sns.color_palette("RdYlGn_r", as_cmap=True)  # 'RdYlGn_r' reverses the palette (green to red)
colors = ["#505050", "#56B4E9","#A066C2"]
colors = ["#939393", "#00427d", "#00652e"]

def plot_error_distribution_by_group(df, error: pd.Series, group_column, title=""):
    # Create a save directory for images
    save_dir = f'images/results/{title}_error_by_group_{group_column}.png'

    # Create a figure with 3 subplots (1 column, 3 rows)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(24, 12), sharex=True)

    # Define group names and their corresponding colors
    if group_column == "name_fluid1":
        groups = ['water', 'TBA', 'camphene']
    else:
        groups = ['Ceramic', 'Metal', 'Polymer']

    # colors = ['blue', 'red', 'green']
    # colors =  ["#505050", "#56B4E9", "#D55E00"]

    # Define custom bin edges with a step of 0.01
    bins = np.arange(-25, 25, 2.5)  # Bins from -0.4 to 0.4 with step size of 0.01

    # Loop through the groups to plot error distribution for each group
    for i, group in enumerate(groups):
        # Filter the dataframe for the current group
        group_df = df[df[group_column] == group]

        # Get the error values for the current group
        group_error = error[group_df.index]  # Use the error values for the current group

        # Plot the histogram on the corresponding subplot
        sns.histplot(group_error, bins=bins, color=colors[i], ax=axes[i], kde=False)

        # Set labels and title for the subplot
        axes[i].set_xlabel("Error", fontsize=28)
        axes[i].set_ylabel("Sample Count", fontsize=28)
        axes[i].tick_params(labelsize=28)
        axes[i].set_title(f"Error Distribution: {group}", fontsize=28)
        axes[i].set_xlim(-25, 25)  # Limit x-axis to match the range

        # Calculate mean and standard deviation for the current group
        mean_error = np.mean(group_error)
        std_error = np.std(group_error)

        # Add mean and std dev as text labels on the plot
        axes[i].text(0.1, 0.85, f'Mean: {mean_error:.1f}%', ha='left', va='center', transform=axes[i].transAxes,
                     fontsize=36)
        axes[i].text(0.1, 0.65, f'STD: {std_error:.0f}%', ha='left', va='center', transform=axes[i].transAxes,
                     fontsize=36)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing
    plt.tick_params(labelsize=28)  # Set font size for tick labels
    # Tight layout to avoid overlap of titles and labels
    plt.tight_layout()
    # Save the plot to the specified directory
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    # Show and save the plot
    plt.show()


def plot_prediction_performance_by_group(true_values, predicted_values, group_column):
    save_dir = f'images/results/perf_by_group.png'

    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'True': true_values,
        'Predicted': predicted_values,
        'Group': group_column
    })

    # Compute absolute error
    error = np.abs(true_values - predicted_values)

    # Define colormap for heatmap
    n_bins = 5
    # cmap = LinearSegmentedColormap.from_list('green_yellow_red', ['darkgreen', 'green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred'], N=n_bins)
    cmap = LinearSegmentedColormap.from_list('green_yellow_red_transparent',
                                             ['#aed476', '#ffff00', '#ff6b6b'], N=n_bins)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(24, 12))

    # Create grid for interpolation
    xi = np.linspace(-100, 105, 50)
    yi = np.linspace(-100, 105, 50)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((df['True'], df['Predicted']), error, (X, Y), method='cubic')

    # Normalize error for color mapping
    norm = Normalize(vmin=0, vmax=np.max(error))

    # Heatmap background
    im = ax.imshow(Z, extent=[-100, 105, -100, 105], origin='lower', cmap=cmap, norm=norm, alpha=0.25, aspect='auto')

    # Get top 3 groups
    top_groups = df['Group'].value_counts().nlargest(3).index.tolist()

    # Define unique markers for each group
    markers = ['^', 's', 'o']

    # Use seaborn palette for colors
    # colors = sns.color_palette("tab10", len(top_groups))  # Extract distinct colors
    # colors = ["#505050", "#56B4E9", "#D55E00"]  # Gray, Light Blue, Purple

    # Plot scatter points for each group
    for i, group in enumerate(top_groups):
        group_df = df[df['Group'] == group]
        ax.scatter(group_df['True'], group_df['Predicted'],
                   color=colors[i],  # Assign color from palette
                   edgecolor='black',
                   linewidths=1.2,
                   marker=markers[i],  # Assign unique marker
                   alpha=0.85,
                   label=group,
                   s=120)  # Marker size

    # Add perfect prediction line (y = x)
    ax.plot([-100, 105], [-100, 105], color='black', linestyle='--', label='Perfect Prediction')

    # Labels and title with fontsize 28
    ax.set_xlabel('True Values', fontsize=28)
    ax.set_ylabel('Predicted Values', fontsize=28)
    # ax.set_title('True vs Predicted Values by Group', fontsize=28)

    # Set axis limits
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    # Set tick labels size
    ax.tick_params(labelsize=28)

    # Colorbar customization
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Prediction Error', fontsize=28)  # Colorbar title
    cbar.ax.tick_params(labelsize=28)  # Colorbar tick labels

    # Add legend and set font size
    ax.legend(title='Groups', title_fontsize=32, fontsize=32, loc='best')

    # Save & display plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    plt.show()

def plot_prediction_performance(true_y, prediction: pd.Series, error: pd.Series, title=""):
    save_path = f'images/results/{title}_perf.png'
    # Red-focused colormap
    # Model performance plot
    fig, ax = plt.subplots(figsize=(24, 12))  # This ensures t he figure and Axes are linked
    sns.scatterplot(x=true_y, y=prediction, hue=error,
                         palette=palette)
    norm = plt.Normalize(0, 40)  # set min and max for color_bar
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    # Adding colorbar correctly
    cbar = ax.figure.colorbar(sm, ax=ax)  # link colorbar to the Axes object
    cbar.set_label("Error", fontsize=28)
    cbar.ax.tick_params(labelsize=28)  # Increase colorbar tick label size

    ax.set_xlabel("True Porosity", fontsize=28)
    ax.set_ylabel(f"Predicted Porosity", fontsize=28)
    ax.get_legend().remove()


    ax.set(ylim=(-10, 111))
    ax.set(xlim=(-10, 111))
    ax.tick_params(labelsize=28)
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()


def plot_error_distribution(df, error: pd.Series, title=""):
    save_dir = f'images/results/{title}_error.png'
    fig, ax = plt.subplots(figsize=(24, 12)) # This ensures the figure and Axes are linked
    sns.histplot(data=df, x=error,  # hue="Group",
                      bins=20,
                      palette=sns.color_palette("hls", 3)
                      )
    ax.set_xlabel("Error", fontsize=20)
    ax.set_ylabel(f"Sample count", fontsize=20)
    ax.tick_params(labelsize=28)
    ax.set(xlim=(-25, 25))
    x_axis = [round(num, 2) for num in np.linspace(-25, 25, 7)]
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    plt.show()



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
df_train[target] = df_train[target] * 100
df_test[target] = df_test[target] * 100

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
    train_preds, test_preds = pipe.predict(df_train)*100, pipe.predict(df_test)*100
    df_test[f'{pipe.name}_prediction'] = test_preds
    mae_test = np.abs(test_preds - df_test[target])
    df_test[f'{pipe.name}_mae'] = mae_test
    print("\n Train results")
    r2_train, mae_train, mse_train, mape_train = utils.get_regression_metrics(train_preds, df_train[target])
    # r2_train, mae_train, mse_train, mape_train = r2_train, round(mae_train, 2), round(mse_train, 2), round(mape_train, 2)
    p_train = r2_score(df_train[target], train_preds)
    formatted_p_train = f"{p_train:.2f}" if p_train >= 0.001 else "<0.001"

    print("Test results")
    r2_test, mae_test, mse_test, mape_test = utils.get_regression_metrics(test_preds, df_test[target])
    # r2_test, mae_test, mse_test, mape_test = r2_test, round(mae_test, 2), round(mse_test, 2), round(mape_test, 2)

    p_test = r2_score(df_test[target], test_preds)
    formatted_p_test = f"{p_test:.2f}" if p_test >= 0.001 else "<0.001"

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

    test_preds_nulls = pipe.predict(df_test_filtered)*100
    r2_test_nulls, mae_test_nulls, mse_test_nulls, mape_test_nulls = utils.get_regression_metrics(test_preds_nulls, df_test[target])
    # r2_test_nulls, mae_test_nulls, mse_test_nulls, mape_test_nulls = r2_test_nulls, round(mae_test_nulls, 2), round(mse_test_nulls, 2), round(mape_test_nulls, 2)

    # Add model results
    result_model_dict = {'name': pipe.name,
                         'r2_train': r2_train, 'p_train': formatted_p_train, 'mae_train': mae_train, 'mse_train': mse_train,
                         'r2_test': r2_test, 'p_test': formatted_p_test, 'mae_test': mae_test, 'mse_test': mse_test}
    results_models_dict_list.append(result_model_dict)
    preds_train, preds_test = pipe.predict(df_train)*100, pipe.predict(df_test)*100
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
        grouped = df_test.groupby(groupby_feature)  # to get metrics
        grouped_train = df_train.groupby(groupby_feature)  # to get most frequent groups
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
            grouped_p = r2_score( group_y_true, group_y_pred)
            formatted_grouped_p = f"{grouped_p:.2f}" if grouped_p >= 0.001 else "<0.001"

            group_variance = round(np.var(group_y_pred), 2)
            group_std = round(np.std(group_y_true), 1)  # true porosity std
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
plt.savefig(f"{im_path}/r2_vs_std.png", dpi=600)
plt.show()


## Performance Plots
selected_model_name = "catb"
prediction = df_test[f"{selected_model_name}_prediction"]*100
true_y = df_test[parser.target]*100
error = df_test[f"{selected_model_name}_prediction"] - df_test[parser.target]
plot_prediction_performance(df_test[parser.target], df_test[f"{selected_model_name}_prediction"], df_test[f"{selected_model_name}_mae"], title=selected_model_name)
plot_error_distribution(df_test, error)
plot_prediction_performance_by_group(df_test[parser.target], df_test[f"{selected_model_name}_prediction"], df_test['material_group'])
# plot_prediction_performance_by_group(df_test[parser.target], df_test[f"{selected_model_name}_prediction"], df_test['name_fluid1'])


# plot_error_distribution_by_group(df_test, error, 'name_fluid1')
plot_error_distribution_by_group(df_test, error, 'material_group')
