from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from grid_search_params import *
from custom_preprocessors import Preprocessors


def get_model_by_name(seed, cat_cols, selected_feats, target, encode_min_frequency=0.1):
    """Select model by number index with associated preprocessing and search space."""

    num_cols = [col for col in selected_feats if col not in cat_cols + [target]]
    # Warning: cuda_1 is for training with GPU, remove it if not configured.
    # Define all model options as (name, model_instance, search_space, preprocessor)
    options = {
        0: (
            "catb_native",
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.opd()
        ),
        1: (
            "catb_native_impute",
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.impute_1hot(
                impute_cols=num_cols,
                cat_cols=cat_cols,
                encode_min_frequency=encode_min_frequency
            )
        ),
        2: (
            "catb_onehot[selected]",
            CatBoostRegressor(random_state=seed, logging_level='Silent'),
            param_grid_catb,
            Preprocessors.onehot(cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        3: (
            "xgb_onehot",
            XGBRegressor(random_state=seed, logging_level='Silent', tree_method="gpu_hist", device="cuda:1"),
            param_grid_xgb,
            Preprocessors.onehot(cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        4: (
            "xgb_impu",
            XGBRegressor(random_state=seed, logging_level='Silent', tree_method="gpu_hist", device="cuda:1"),
            param_grid_xgb,
            Preprocessors.impute_1hot(
                impute_cols=num_cols,
                cat_cols=cat_cols,
                encode_min_frequency=encode_min_frequency
            )
        ),
        5: (
            "lr",
            LinearRegression(),
            param_grid_lr,
            Preprocessors.impute_1hot(
                impute_cols=[col for col in selected_feats if col not in cat_cols + [target]],
                cat_cols=[col for col in cat_cols if col in selected_feats],
                encode_min_frequency=encode_min_frequency
            )
        ),
        6: (
            "rf",
            RandomForestRegressor(random_state=seed),
            param_grid_rf,
            Preprocessors.impute_1hot(impute_cols=num_cols, cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)
        ),
        7: (
            "nn",
            MLPRegressor(random_state=seed),
            param_grid_nn,
            Preprocessors.impute_1hot(impute_cols=num_cols, cat_cols=cat_cols, encode_min_frequency=0.01)
        ),
    }

    # Display model choices
    print("\nAvailable models:")
    for k, (name, _, _, _) in options.items():
        print(f"  [{k}] {name}")

    # Select model
    try:
        choice = int(input("\nSelect model by number: "))
        model_name, model, search_space, selected_preprocessor = options[choice]
    except (ValueError, KeyError):
        print("Invalid selection. Exiting.")
        exit()

    # Special case for linear regression: restrict features to 'vf_total'
    if model_name == "lr":
        selected_feats[:] = [col for col in selected_feats if col == "vf_total"]

    print("Model selected:", model_name)
    return model, search_space, selected_preprocessor, model_name
