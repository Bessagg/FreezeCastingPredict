import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import shutil
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

def save_list_to_txt(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write('%s\n' % item)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_regression_metrics(preds, y, opt_print=True, round_n=2):
    r2 = "{:.02f}".format(r2_score(y, preds))
    mse = "{:.03f}".format(round(mean_squared_error(y, preds), round_n))
    mae = "{:.03f}".format(round(mean_absolute_error(y, preds), round_n))
    mape = "{:.03f}".format(round(mean_absolute_percentage_error(y, preds), round_n))

    if opt_print:
        print('r2:', r2_score(y, preds))
        print(f"mse:", mse, "mae", mae, 'mape', mape)
    return r2, mae, mse, mape


def save_pipeline_reg(clf, seed, models_folder, model_name, train_name, r2_train, r2, mae, mse):
    now = datetime.now().strftime('%H%M%S')
    model_filename = f"{model_name}-{train_name}-{now}-{seed}-r{r2_train}-r{r2}-ma{mae}-ms{mse}"
    model_results_path = f"temp_models/{models_folder}/{model_name}/{model_filename}"
    model_fullpath = f"{model_results_path}/model/"
    if not os.path.isdir(model_fullpath):
        os.makedirs(model_fullpath)

    # If model is single save with pickle
    if clf.preprocessor_name.__contains__('single'):
        pickle.dump(clf, open(f"{model_fullpath}/{model_filename}.pickle", 'wb'))
        print(f"Model saved with PICKLE at: {model_fullpath}/{model_filename}")
    # Else save with joblib
    else:
        joblib.dump(clf, f"{model_fullpath}/{model_filename}.pickle")
        print(f"Model saved with JOBLIB at: {model_fullpath}/{model_filename}")
    return model_results_path, model_filename


def save_artifacts(model_results_path, selected_feats=None, data_parser_filepath=None):
    # Save selected cols to .txt
    if selected_feats:
        save_list_to_txt(selected_feats, f"{model_results_path}/selected_feats.txt")
        print("saved selected cols to", f"{model_results_path}/selected_feats.txt")
    # Save dataparser
    if data_parser_filepath:
        shutil.copy2(data_parser_filepath, f"{model_results_path}/{Path(data_parser_filepath).name}")
        print("saved data_parser file to", f"{model_results_path}/selected_feats.txt")


def load_pipeline(model_path):
    with open(model_path, 'rb') as f:
        pipeline = joblib.load(model_path)
    return pipeline


def handle_outliers(df, col_name='vf_total', threshold=3):
    """
    Removes outliers in a column, ignoring rows with NaN values.

     Args:
        df: The pandas DataFrame.
        col_name: The name of the column to handle (default: 'vf_total').
        threshold: The Z-score threshold for outliers (default: 3). 99.7% in a normal dist
     Returns:
        A new DataFrame with outliers filtered.
    """
    # Select rows without NaN in 'vf_total' (assuming you want to keep them)
    df_filtered = df[df[col_name].notna()]

    # Calculate mean and standard deviation for 'vf_total' (on non-NaN rows)
    mean_val = df_filtered[col_name].mean()
    std_dev = df_filtered[col_name].std()

    # Calculate Z-scores
    z_scores = (df_filtered[col_name] - mean_val) / std_dev

    # Filter out outliers based on threshold
    df_filtered = df_filtered[abs(z_scores) <= threshold]

    # Combine the filtered data (without outliers) with the original DataFrame (including NaN rows)
    return pd.concat([df[df[col_name].isna()], df_filtered])
