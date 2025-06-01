from typing import List, Dict, Optional

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from analysis_model import safe_round_to_int
from helpers.utils import mean_absolute_percentage_error


def get_regression_metrics(preds, y, opt_print=False, round_n=2):
    r2 = "{:.02f}".format(r2_score(y, preds))
    mse = "{:.03f}".format(round(mean_squared_error(y, preds), round_n))
    mae = "{:.03f}".format(round(mean_absolute_error(y, preds), round_n))
    mape = "{:.03f}".format(round(mean_absolute_percentage_error(y, preds), round_n))

    if opt_print:
        print('r2:', r2_score(y, preds))
        print(f"mse:", mse, "mae", mae, 'mape', mape)
    return r2, mae, mse, mape


def get_test_metrics_by_group(
    df_test: pd.DataFrame,
    test_preds: pd.Series,
    target: str,
    group_cols: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns test regression metrics grouped by specified categorical columns.

    Parameters:
    - df_test: test dataframe with features and target
    - test_preds: predictions aligned with df_test index
    - target: target column name in df_test
    - group_cols: list of categorical columns to group by

    Returns:
    - Dictionary of dictionaries:
        {
          'group_col': {
              'group_val': {
                  'count': int,
                  'r2': float,
                  'mae': float,
                  'mse': float,
                  'mape': float
              },
              ...
          },
          ...
        }
    """
    grouped_metrics = {}
    for col in group_cols:
        metrics_by_value = {}
        unique_vals = df_test[col].dropna().unique()
        for val in sorted(unique_vals):
            group = df_test[df_test[col] == val]
            preds_group = test_preds[group.index]
            r2, mae, mse, mape = get_regression_metrics(preds_group, group[target])
            metrics_by_value[val] = {
                'count': len(group),
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'mape': mape
            }
        grouped_metrics[col] = metrics_by_value

    return grouped_metrics


def compute_overall_metrics(pipeline, df: pd.DataFrame, target: str) -> Dict:
    """
    Predict and compute regression metrics on a single dataframe.
    Returns a dict with r2, mae, mse, mape, and formatted p-value.
    """
    preds = pipeline.predict(df)
    true = df[target]
    r2, mae, mse, mape = get_regression_metrics(preds, true)
    r2, mae, mse, mape = [safe_round_to_int(x) for x in [r2, mae, mse, mape]]

    p_val = r2_score(true, preds)
    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "mape": mape,
    }


def compute_group_metrics(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str,
        group_feature: str, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Compute regression metrics per category of a group feature for both train and test sets.

    Parameters:
        df_train (pd.DataFrame): DataFrame with train true and predicted values.
        df_test (pd.DataFrame): DataFrame with test true and predicted values.
        target (str): Name of the target column (true values).
        group_feature (str): Feature to group by.
        top_n (int, optional): Top N most frequent categories to evaluate. If None, use all.

    Returns:
        pd.DataFrame: Metrics for each category in train/test split.
    """
    counts = df_train[group_feature].value_counts() + df_test[group_feature].value_counts()
    categories = counts.index if top_n is None else counts.nlargest(top_n).index
    records = []

    for cat in categories:
        record = {group_feature: cat}

        for split_name, df in [("train", df_train), ("test", df_test)]:
            mask = df[group_feature] == cat
            if mask.sum() == 0:
                record[f"{split_name}_r2"] = None
                record[f"{split_name}_mae"] = None
                record[f"{split_name}_mse"] = None
                record[f"{split_name}_mape"] = None
                record[f"{split_name}_count"] = 0
                continue

            y_true = df.loc[mask, target]
            y_pred = df.loc[mask, "prediction"]
            r2, mae, mse, mape = [safe_round_to_int(x) for x in get_regression_metrics(y_pred, y_true, opt_print=False)]
            record[f"{split_name}_r2"] = r2
            record[f"{split_name}_mae"] = mae
            record[f"{split_name}_mse"] = mse
            record[f"{split_name}_mape"] = mape
            record[f"{split_name}_count"] = mask.sum()

        records.append(record)
    return pd.DataFrame.from_records(records)
