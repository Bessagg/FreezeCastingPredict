from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from grid_search_params import *
from custom_preprocessors import Preprocessors
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor


def get_model_by_name(seed, cat_cols, selected_feats, target, encode_min_frequency=0.1):
    """Select Model"""
    model_name = input('\n Type model name: [catb, catb2, xgb, lr, rf, nn]\n')
    if model_name.lower() == "catb":
        # model = CatBoostRegressor(logging_level='Silent', random_state=seed)
        model = CatBoostRegressor(random_state=seed, logging_level='Silent')
        search_space = param_grid_catb
        selected_preprocessor = Preprocessors.onehot(cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)

    elif model_name.lower() == "catb2":
        "catboost with native categorical handling"
        # model = CatBoostRegressor(logging_level='Silent', random_state=seed)
        model = CatBoostRegressor(random_state=seed, logging_level='Silent')  # n_jobs=-1, tree_method="gpu_hist",
        search_space = param_grid_catb
        selected_preprocessor = Preprocessors.opd()

    elif model_name.lower() == "xgb":
        model = XGBRegressor(random_state=seed, logging_level='Silent', tree_method="gpu_hist", device="cuda:1")
        search_space = param_grid_xgb
        selected_preprocessor = Preprocessors.onehot(cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)

    elif model_name.lower() == "lr":
        model = LinearRegression()
        search_space = param_grid_lr
        selected_feats[:] = [col for col in selected_feats if col == "vf_total"]
        # only consider vf_total
        selected_preprocessor = Preprocessors.impute_1hot(
            impute_cols=[element for element in selected_feats if element not in cat_cols + [target]],
            cat_cols=[col for col in cat_cols if col in selected_feats],
            encode_min_frequency=encode_min_frequency)

    elif model_name.lower() == "rf":
        model = RandomForestRegressor(random_state=seed)
        search_space = param_grid_rf
        selected_preprocessor = Preprocessors.impute_1hot(impute_cols=[element for element in selected_feats if element not in cat_cols + [target]],
                                                          cat_cols=cat_cols, encode_min_frequency=encode_min_frequency)

    elif model_name.lower() == "nn":
        model = MLPRegressor(random_state=seed)  # Create MLPRegressor object
        search_space = param_grid_nn
        selected_preprocessor = Preprocessors.impute_1hot(
            impute_cols=[element for element in selected_feats if element not in cat_cols + [target]],
            cat_cols=cat_cols, encode_min_frequency=0.01)

    else:
        model = None
        search_space = None
        print("did not find a model for model name:", model_name)
        exit()
    print("Model name:", model_name)
    return model, search_space, selected_preprocessor, model_name
