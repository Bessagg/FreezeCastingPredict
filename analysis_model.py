import pandas as pd
from data_parser import DataParser
from helpers import utils
import warnings
import glob
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 600)

"""General Setup"""
print("Parsing data...")
parser = DataParser()
target = parser.target

"""Fetch dataset"""
print("Loading Data...")
selected_feats_name = "all_cols"
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
    pipe_ggparent_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pipe_path))))
    train_preds, test_preds = pipe.predict(df_train), pipe.predict(df_test)
    r2_train, mae_train, mse_train = utils.get_regression_metrics(train_preds, df_train[target])
    r2_test, mae_test, mse_test = utils.get_regression_metrics(test_preds, df_test[target])

    # Add model results
    result_model = {'name': pipe.name, 'r2_train': r2_train, 'mae_train': mae_train, 'mse_train': mse_train,
                    'r2_test': r2_test, 'mae_test': mae_test, 'mse_test': mse_test}
    results_models.append(result_model)
    preds_train, preds_test = pipe.predict(df_train), pipe.predict(df_test)
    results_train = {'name': pipe.name, 'prediction_train': preds_train, 'prediction_test': preds_test,
                    'train_true': df_train[target], 'test_true': df_test[target],
                    'model_type': '-', }

    # Add importances if available
    estimator = pipe[-1]
    if hasattr(estimator, 'feature_importances_'):
        feature_importances = estimator.feature_importances_
        for i, importance in enumerate(feature_importances):
            results_importances.append({
                'model': pipe.name,
                'feature_index': i,
                'feature_importance': importance
            })

    # Calculate metrics grouped by the specified feature
    groupby_features = ['material_group', 'name_part1', 'name_fluid1']
    for groupby_feature in groupby_features:
        grouped = df_test.groupby(groupby_feature)
        for group_name, group_indices in grouped.groups.items():
            group_y_true = df_test[target][group_indices]
            group_y_pred = preds_test[group_indices]
            grouped_r2, grouped_mae, grouped_mse = (
                utils.get_regression_metrics(group_y_pred, group_y_true, opt_print=False))
            # append results
            results_groups.append({
                'model': pipe.name,
                'groupby_feature': groupby_feature,
                'group': group_name,
                'r2': grouped_r2,
                'mae': grouped_mae,
                'mse': grouped_mse
            })

