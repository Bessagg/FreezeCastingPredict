from typing import List, Dict

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    group_cols: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
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
