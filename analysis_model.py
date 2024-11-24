import pandas as pd
from data_parser import DataParser
from helpers import utils
import warnings
import glob
import os
from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr, chi2_contingency
from scipy.stats import chi2_contingency, kruskal

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 600)

im_path = 'images/model_analysis'


def plot_prediction_performance(true_y, prediction: pd.Series, error: pd.Series, title=str):
    pallete = sns.color_palette("hls", 3)
    # Model performance plot
    plt.figure(figsize=(18, 12))
    ax = sns.scatterplot(x=true_y, y=prediction, hue=error,
                         palette=pallete)
    norm = plt.Normalize(0, 0.4)  # set min and max for color_bar
    sm = plt.cm.ScalarMappable(cmap=pallete, norm=norm)
    sm.set_array([])
    ax.set_xlabel("True Porosity", fontsize=20)
    ax.set_ylabel(f"Predicted Porosity - {title}", fontsize=20)
    ax.get_legend().remove()
    ax.figure.colorbar(sm)
    ax.set(ylim=(0.01, 1.01))
    ax.set(xlim=(0.01, 1.01))
    ax.tick_params(labelsize=20)
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.show()
    plt.savefig(f'images/results/{title}_perf', bbox_inches='tight')


def plot_error_distribution(df, error: pd.Series, title=str):
    plt.figure(figsize=(18, 12))
    bx = sns.histplot(data=df, x=error,  # hue="Group",
                      bins=20,
                      palette=sns.color_palette("hls", 3))
    bx.set_xlabel("Error", fontsize=20)
    bx.set_ylabel(f"Sample count - {title} - seed 42", fontsize=20)
    bx.tick_params(labelsize=20)
    bx.set(xlim=(-0.4, 0.4))
    x_axis = [round(num, 2) for num in np.linspace(-0.4, 0.4, 7)]
    plt.show()
    plt.savefig(f'images/results/{title}_error', bbox_inches='tight')


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
    print("Train results")
    r2_train, mae_train, mse_train, mape_train = utils.get_regression_metrics(train_preds, df_train[target])
    print("Test results")
    r2_test, mae_test, mse_test, mape_test = utils.get_regression_metrics(test_preds, df_test[target])
    print("Test nulls results")
    df_test_nulls = df_test.copy()
    for col in df_test_nulls.columns:
        if col not in ['name_part1', 'name_fluid1', 'vf_total', 'material_group', target]:
            if col in parser.all_num_cols:
                df_test_nulls[col] = np.nan
            else:
                df_test_nulls[col] = ""
    test_preds_n = pipe.predict(df_test_nulls)
    r2_test_nulls, mae_test_nulls, mse_test_nulls, mape_test_nulls = utils.get_regression_metrics(test_preds_n, df_test[target])

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
