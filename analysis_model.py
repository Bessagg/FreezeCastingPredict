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
results_models = []
results_importances = []
results_train = pd.DataFrame()
results_test = pd.DataFrame()
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
    r2_test_n, mae_test_n, mse_test_n, mape_test_n = utils.get_regression_metrics(test_preds_n, df_test[target])

    # Add model results
    result_model = {'name': pipe.name, 'r2_train': r2_train, 'mae_train': mae_train, 'mse_train': mse_train,
                    'r2_test': r2_test, 'mae_test': mae_test, 'mse_test': mse_test}
    results_models.append(result_model)
    preds_train, preds_test = pipe.predict(df_train), pipe.predict(df_test)
    results_train = {'name': pipe.name, 'feats': pipe.feats_name, 'prediction_train': preds_train, 'prediction_test': preds_test,
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
df_groups_filtered = df_groups.query("model == 'catb' and top5 == True").sort_values(by=['groupby_feature', 'train samples'], ascending=False)
print("Selected model results by group:")
print(df_groups_filtered)


print("Relationship between R2 and STD")
df_groups_filtered['variance'] = df_groups_filtered['r2'].astype(float)
df_groups_filtered['std'] = df_groups_filtered['std'].astype(float)
plt.scatter(df_groups_filtered['r2'].astype(float), df_groups_filtered['std'].astype(float))
plt.xlabel(r'$R^2$ value')
plt.ylabel("STD")
from sklearn.linear_model import LinearRegression
X = df_groups_filtered[['std']].values.reshape(-1, 1)  # Independent variable
# X = X.flatten().astype(float).tolist()
y = df_groups_filtered['r2'].values  # Dependent variable
y = y.astype(float)
# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)
# Predict y values
y_pred = model.predict(X)
# Calculate the R^2 score
r2 = r2_score(y, y_pred)
# Plot the results
plt.clf()
plt.scatter(X, y, color='blue', label='Grouped data')
plt.plot(X, y_pred, color='red', linewidth=2, label=f'Linear regression (R² = {r2:.2f})')
plt.xlabel('Standard Deviation of true porosity')
plt.ylabel(r'$R^2$ Value')
plt.title('Linear Regression between STD and $R^2$')
plt.legend()
plt.show()
plt.savefig(f"{im_path}/r2_vs_std.png")
