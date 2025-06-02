from typing import List, Dict, Optional
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from helpers.utils import mean_absolute_percentage_error


def get_regression_metrics(preds, y, opt_print=False, round_n=2):
    r2 = "{:.02f}".format(r2_score(y, preds))
    #     mse = "{:.03f}".format(round(mean_squared_error(y, preds), round_n))
    mse = "{:.03f}".format(mean_squared_error(y, preds))
    mae = "{:.03f}".format(mean_absolute_error(y, preds))
    mape = "{:.03f}".format(mean_absolute_percentage_error(y, preds))

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
    # r2, mae, mse, mape = [safe_round_to_int(x) for x in [r2, mae, mse, mape]]

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
            r2, mae, mse, mape = [x for x in get_regression_metrics(y_pred, y_true, opt_print=False)]
            record[f"{split_name}_r2"] = r2
            record[f"{split_name}_mae"] = mae
            record[f"{split_name}_mse"] = mse
            record[f"{split_name}_mape"] = mape
            record[f"{split_name}_count"] = mask.sum()

        records.append(record)
    return pd.DataFrame.from_records(records)


def get_pipeline_metrics(pipelines: List, df_train: pd.DataFrame, df_test: pd.DataFrame, target: str,
                         features_to_analyze: List[str], columns_not_null: List[str], selected_pipelines=None) -> Dict[str, object]:
    """
    Evaluate multiple pipelines on various metrics.
    Returns:
        - 'overall_metrics': DataFrame for all pipelines
        - 'no_nan_metrics': DataFrame for selected_pipelines no-NaN subset
        - 'only_nans_metrics': DataFrame for selected_pipelines only-NaNs subset
        - 'group_metrics': nested dict {pipeline: {feature: {'topn': df, 'all': df}}}
    """
    if selected_pipelines is None:
        selected_pipelines = ["catb"]

    overall = []
    group = {}
    no_nan, only_nans = [], []

    for pipe in pipelines:
        # Predict and scale if necessary
        pred_train = pipe.predict(df_train)
        pred_test = pipe.predict(df_test)

        # # If most values are in [0, 1], assume it's not in percentage and scale
        # if (pred_train < 1.5).mean() > 0.9 and (pred_test < 1.5).mean() > 0.9:
        #     pred_train *= 100
        #     pred_test *= 100
        #     if df_train[target].max() < 1.5 and df_test[target].max() < 1.5:
        #         df_train[target] *= 100
        #         df_test[target] *= 100


        # Add predictions to copies of the DataFrames
        df_train_copy = df_train.copy()
        df_test_copy = df_test.copy()
        df_train_copy["prediction"] = pred_train
        df_test_copy["prediction"] = pred_test

        # Overall metrics
        keys = ['r2', 'mae', 'mse', 'mape']
        tr =   dict(zip(keys, get_regression_metrics(pred_train, df_train_copy[target], opt_print=False)))
        te =   dict(zip(keys, get_regression_metrics(pred_test, df_test_copy[target], opt_print=False)))
        overall.append({
            'pipeline': pipe.name,
            **{f'train_{k}': v for k, v in tr.items()},
            **{f'test_{k}': v for k, v in te.items()}
        })

        # Only for selected pipelines
        if pipe.name in selected_pipelines:
            for mode in ['no_nan', 'only_nans']:
                subset = filter_null_mask_by_type(df_test_copy, columns_not_null, pipe.selected_feats, mode)
                if not subset.empty:
                    m =  compute_overall_metrics(pipe, subset, target)
                    entry = {
                        'pipeline': pipe.name,
                        'count': len(subset),
                        **{f'{mode}_{k}': v for k, v in m.items()}
                    }
                    (no_nan if mode == 'no_nan' else only_nans).append(entry)

            # Group-wise metrics
            feat_dict = {}
            for feat in features_to_analyze:
                feat_dict[feat] = {}
                feat_dict[feat]["topn"] =  compute_group_metrics(df_train_copy, df_test_copy, target, feat, top_n=5)
                feat_dict[feat]["all"] =   compute_group_metrics(df_train_copy, df_test_copy, target, feat)
            group = feat_dict

    return {
        'overall_metrics': pd.DataFrame(overall),
        'no_nan_metrics': pd.DataFrame(no_nan),
        'only_nans_metrics': pd.DataFrame(only_nans),
        'group_metrics': group
    }


def print_deepest_keys(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            print_deepest_keys(v, path + (k,))
    elif isinstance(d, list):
        for i, item in enumerate(d):
            print_deepest_keys(item, path + (f"[{i}]",))
    else:
        print(" -> ".join(path))
        print(d)
        print('-' * 60)


def safe_round_to_int(x):
    if pd.isna(x):
        return x
    try:
        return int(round(float(x)))
    except Exception:
        return x


def filter_topn_categories(df, column_name, n=3):
    top3_group = df[column_name].value_counts().iloc[:n].index.to_list()
    return df[(df[column_name].isin(top3_group))]


def filter_null_mask_by_type(df: pd.DataFrame, columns_not_null: List[str], selected_feats: List[str], filter_type: str) -> pd.DataFrame:
    """
    Filters DataFrame based on null values in selected features:
    - 'no_nan': Rows where all selected features are non-null AND columns_not_null are non-null.
    - 'only_nans': Rows where some selected features are null, but columns_not_null are non-null.
                   Also ensures there is at least one NaN in selected_feats to qualify.
    """
    # First, ensure all 'columns_not_null' are non-null for both types
    df_base = df[df[columns_not_null].notna().all(axis=1)].copy()

    if filter_type == 'no_nan':
        # All selected features must be non-null
        return df_base[df_base[selected_feats].notna().all(axis=1)]
    elif filter_type == 'only_nans':
        # At least one selected feature is non-null (to ensure the row is relevant to the model)
        # AND at least one selected feature is null
        has_some_nans = df_base[selected_feats].isna().any(axis=1)
        has_some_non_nans = df_base[selected_feats].notna().any(axis=1)
        return df_base[has_some_nans & has_some_non_nans] # Includes rows where some are nan, some are not
    else:
        raise ValueError("filter_type must be 'no_nan' or 'only_nans'")
