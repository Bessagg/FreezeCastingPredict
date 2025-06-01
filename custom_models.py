from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from typing import Union, Optional
from grid_search_params import *
from custom_preprocessors import Preprocessors


def get_model_by_name(seed, cat_cols, selected_feats, target, encode_min_frequency=0.1, model_name: Optional[Union[str, int]] = None
):
    """
      Select a model configuration by name or index.

      Parameters:
      - seed (int): Random seed for reproducibility.
      - cat_cols (list): List of categorical feature names.
      - selected_feats (list): List of selected feature names.
      - target (str): Target column name.
      - encode_min_frequency (float): Minimum frequency for encoding categories.
      - model_name (str | int | None):
          - If None, prompts user to select a model.
          - If int, selects model by index from predefined options.
          - If str, selects model by its string name.

      Returns:
      - model: Estimator instance.
      - search_space: Hyperparameter grid.
      - selected_preprocessor: Associated preprocessor object.
      - model_name (str): Selected model's name.
      """
    # Special case for LR
    if model_name == "lr_solidloading":
        selected_feats = ['vf_total']
        cat_cols = []
    num_cols = [col for col in selected_feats if col not in cat_cols + [target]]

    options = {
        1: (
            "catb_native",  # onehot not used
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.opd()
        ),
        2: (
            "catb_native_impute", # not working, not even with correct preprocessor. catb is a pain with sklearn pipelines.
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.opd()
        ),
        4: (
            "catb_onehot[selected]",  # onehot used
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.onehot(cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        5: (
            "catb_onehot_impute",  # onehot used
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.impute_1hot(impute_cols=num_cols,
                                      cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        6: (
            "xgb_onehot",
            XGBRegressor(random_state=seed, logging_level='Silent', tree_method="gpu_hist", device="cuda:1"),
            param_grid_xgb,
            Preprocessors.onehot(cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        7: (
            "xgb_impute",
            XGBRegressor(random_state=seed, logging_level='Silent', tree_method="gpu_hist", device="cuda:1"),
            param_grid_xgb,
            Preprocessors.impute_1hot(
                impute_cols=num_cols,
                cat_cols=cat_cols,
                encode_min_frequency=encode_min_frequency
            )
        ),
        8: (
            "rf",
            RandomForestRegressor(random_state=seed),
            param_grid_rf,
            Preprocessors.impute_1hot(impute_cols=num_cols, cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        9: (
            "lr",
            LinearRegression(),
            param_grid_lr,
            Preprocessors.impute_1hot(
                impute_cols=[col for col in selected_feats if col not in cat_cols + [target]],
                cat_cols=[col for col in cat_cols if col in selected_feats],
                encode_min_frequency=encode_min_frequency
            )
        ),
        10: (
            "lr_solidloading",
            LinearRegression(),
            param_grid_lr,
            Preprocessors.impute(num_cols)
        ),

        0: (
            "nn",
            MLPRegressor(random_state=seed),
            param_grid_nn,
            Preprocessors.impute_1hot(impute_cols=num_cols, cat_cols=cat_cols, encode_min_frequency=0.01)
        ),
    }

    model_lookup = {v[0]: v for v in options.values()}

    if model_name is None:
        # Print options and ask user
        print("\nAvailable models:")
        for k, (name, _, _, _) in options.items():
            print(f"  [{k}] {name}")
        try:
            choice = int(input("\nSelect model by number: "))
            model_name, model, search_space, selected_preprocessor = options[choice]
        except (ValueError, KeyError):
            print("Invalid selection. Exiting.")
            exit()
    elif isinstance(model_name, int) and model_name in options:
        model_name, model, search_space, selected_preprocessor = options[model_name]
    elif isinstance(model_name, str) and model_name in model_lookup:
        model_name, model, search_space, selected_preprocessor = model_lookup[model_name]
    else:
        raise ValueError(f"Invalid model_name: {model_name}")



    print("Model selected:", model_name)
    return model, search_space, selected_preprocessor, model_name