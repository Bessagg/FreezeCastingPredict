import time
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from helpers import utils, shap_explainer, custom_features
from data_parser import DataParser
import custom_models
from data.additional_samples import df_gpt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 600)


def train_model(model_name=None, feats_name=None, seed=6, cv=5, encode_min_frequency=0.04, shap_opt=None):
    """
      Train a regression model pipeline with preprocessing, grid search, and optional SHAP explanation.

      Parameters:
      -----------
      model_name : str or int, optional
          Model name or index to select from predefined options. If None, prompts user input.
      feats_name : str or int, optional
          Feature set name or index. If None, prompts user input from predefined sets.
      seed : int, default=6
          Random seed for reproducibility.
      cv : int, default=5
          Number of cross-validation folds.
      encode_min_frequency : float, default=0.04
          Minimum frequency threshold for encoding categorical features.
      shap_opt : int, optional
          If 1, runs SHAP explanation. If 0, skips it. If None, prompts user.

      Returns:
      --------
      Saves trained model pipeline, performance metrics, and optionally SHAP plots to disk.
      Prints training and test evaluation results.
      """

    print("Parsing data...")
    parser = DataParser()
    target = parser.target

    feats_options = {
        1: "all_feats",
        2: "reduced_feats",
        3: "simplest_feats"
    }

    if feats_name is None:
        print("\nSelect feature set:")
        for k, v in feats_options.items():
            print(f"{k}: {v}")
        while True:
            try:
                selection = int(input("Enter a number [1-3]: "))
                feats_name = feats_options[selection]
                break
            except (ValueError, KeyError):
                print("Invalid selection. Please enter 1, 2, or 3.")

    selected_feats = parser.feats_dict[feats_name]
    selected_feats.remove(target)

    num_cols = [x for x in parser.all_num_cols if x in selected_feats and x != target]
    cat_cols = [x for x in parser.all_cat_cols if x in selected_feats and x != target]
    ignored_feats = [feat for feat in selected_feats if feat not in set(num_cols + cat_cols)]

    try:
        df_train = pd.read_csv(f"data/{feats_name}/train.csv")
        df_test = pd.read_csv(f"data/{feats_name}/test.csv")
        df_train, df_test = custom_features.add_material_novelty_feature(df_train, df_test, min_count=6)
        df_train, df_test = custom_features.add_bin_material_frequency(df_train, df_test, feature='name_part1')
    except FileNotFoundError:
        print(f'train and test file not found in data/{feats_name}')
        sys.exit(1)

    model, search_space, selected_preprocessor, model_name = custom_models.get_model_by_name(
        seed, cat_cols, selected_feats, target, encode_min_frequency, model_name
    )

    if model_name == "lr_solidloading":
        print(model_name, "selected. Removing other features except Solid Loading")
        selected_feats = 'vf_total'

    selected_preprocessor.fit(df_train[selected_feats], df_train[target])
    pipe = Pipeline([
        ('preprocess', selected_preprocessor),
        ('model', model)
    ])

    print("-"*40)
    print("Selected model:", model_name)
    print("Search space:", search_space)
    print("Fitting Grid...")
    start = time.time()
    if model_name.startswith('catb_native'):
        cat_feature_indices = [df_train[selected_feats].columns.get_loc(col) for col in cat_cols]
        df_train[cat_cols] = df_train[cat_cols].fillna("None")
        df_test[cat_cols] = df_test[cat_cols].fillna("None")
        grid = GridSearchCV(pipe, search_space, cv=cv, scoring='r2', verbose=0)
        grid.fit(df_train[selected_feats], df_train[target], model__cat_features=cat_feature_indices)
    else:
        grid = GridSearchCV(pipe, search_space, cv=cv, scoring='r2', verbose=0)
        grid.fit(df_train[selected_feats], df_train[target])

    print("CV Results\n", pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False))
    print("Best Params\n", grid.best_params_)

    best_clf = grid.best_estimator_
    best_clf.name = model_name
    best_clf.cat_cols = cat_cols
    best_clf.num_cols = num_cols
    best_clf.feats_name = feats_name
    best_clf.cv_results_ = grid.cv_results_
    best_clf.selected_feats = selected_feats
    best_clf.preprocessor_name = selected_preprocessor.name

    estimator = grid.best_estimator_[-1]
    preprocessor = grid.best_estimator_[0]
    print("Elapsed", (time.time() - start) / 60, 'min')

    train_preds = best_clf.predict(df_train[selected_feats])
    test_preds = best_clf.predict(df_test[selected_feats])

    print("Test Results by Group")
    utils.print_test_metrics_by_group(df_test, test_preds, target, ['name_part1_freq_bin', 'year'])

    print("Test GPT Data")
    gpt_preds = best_clf.predict(df_gpt[selected_feats]) * 100
    utils.get_regression_metrics(gpt_preds, df_gpt[target])
    utils.print_test_metrics_by_group(df_gpt, gpt_preds, target, ['name_part1'])

    print("Training Results")
    r2_train, mae_train, mse_train, mape_train = utils.get_regression_metrics(train_preds, df_train[target])
    print(f"R²: {r2_train}, MAE: {mae_train}, MSE: {mse_train}, MAPE: {mape_train}")

    print("\nTest Results")
    r2, mae, mse, mape = utils.get_regression_metrics(test_preds, df_test[target])
    print(f"R²: {r2}, MAE: {mae}, MSE: {mse}, MAPE: {mape}")

    models_folder = f"pipe/{feats_name}"
    model_results_path, model_filename = utils.save_pipeline_reg(
        best_clf, seed, models_folder, model_name, selected_preprocessor.name,
        r2_train, r2, mae, mse
    )

    cols_p = np.char.replace(np.array(preprocessor.get_feature_names_out(), dtype=str), 'remainder__', '')
    train_X_p = pd.DataFrame(preprocessor.transform(df_train[selected_feats]), columns=cols_p)
    test_X_p = pd.DataFrame(preprocessor.transform(df_test[selected_feats]), columns=cols_p)
    print("Count of created preprocessed columns:", len(cols_p))

    if shap_opt is None:
        run_shap = input("Run SHAP? [1 or 0]: ")
        shap_opt = int(run_shap)

    if shap_opt == 1:
        prevalence = df_train[target].mean()
        predicted_label = best_clf.predict(df_test)

        explainer, preprocessed_X, shap_values, shap_confusion_matrix_dict = (
            shap_explainer.get_shap_plotter_inputs(
                estimator, preprocessor, df_test,
                selected_feats, predicted_label, prevalence,
                df_test[target])
        )

        shap_plotter = shap_explainer.ShapPlotter(
            explainer, preprocessed_X, shap_values,
            shap_confusion_matrix_dict,
            col_rename_dict=parser.col_rename_dict,
            save_dirpath=model_results_path + "/shap",
            dataframe_subdir_name="test",
            plots_title=f'{model_name}_test'
        )
        shap_explainer.save_explainer(explainer, shap_plotter.filepath_explainer)
        shap_plotter.run_all_plots()
        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list(), color_feature="Solid Loading")
        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list(), color_feature="Temp Sinter")
        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list(), color_feature="Disp. wf.")
        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list(), color_feature="Solid Diameter")
        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list(), color_feature="Temp. Cold")


        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list(), color_feature="material_group_Polymer")
        shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list())

    print("Finished")


if __name__ == "__main__":
    shap_opt=1
    """Train single pipeline"""
    # train_model()

    """Running all possible pipeline trains by index"""
    if __name__ == "__main__":
        model_names =  ["catb_native", "catb_native_impute", "catb_onehot[selected]",
            "xgb_onehot", "xgb_impu",
            "lr", "lr_solidloading","rf"]
        for model_name in model_names:
            print(f"\n=== Training with model: {model_name} ===")
            train_model(model_name=model_name, feats_name="reduced_feats", shap_opt=shap_opt)
