import time
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import helpers.metrics_utils
from helpers import utils, shap_explainer, custom_features
from data_parser import DataParser
import custom_models
from data.additional_samples import df_additional_data
from analysis_model import FEATURES_TO_ANALYZE, COLUMNS_NOT_NULL

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 600)


def train_model(model_name=None, feats_name=None, seed=6, cv=5, encode_min_frequency=0.04, shap_opt=None, selected_feats=None):
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
        print("\nSelect feature dataset:")
        for k, v in feats_options.items():
            print(f"{k}: {v}")
        while True:
            try:
                selection = int(input("Enter a number [1-3]: "))
                feats_name = feats_options[selection]
                break
            except (ValueError, KeyError):
                print("Invalid selection. Please enter 1, 2, or 3.")

    if selected_feats:
        print("Using provided selected features:", selected_feats)
        selected_feats = selected_feats
    else:         # Fetch selected_feats from parser dict
        selected_feats = parser.feats_dict[feats_name]

    if target in selected_feats:
        selected_feats.remove(target)
    if model_name == 'lr_solidloading':
        print("Using only Solid Loading feature for lr_solidloading model")
        selected_feats = ['vf_total']

    num_cols = [x for x in parser.all_num_cols if x in selected_feats and x != target]
    cat_cols = [x for x in parser.all_cat_cols if x in selected_feats and x != target]
    ignored_feats = [feat for feat in selected_feats if feat not in set(num_cols + cat_cols)]

    try:
        df_train = pd.read_csv(f"data/{feats_name}/train.csv")
        df_test = pd.read_csv(f"data/{feats_name}/test.csv")
        df_train, df_test = custom_features.add_material_novelty_feature(df_train, df_test, min_count=5)
        df_train, df_test = custom_features.add_bin_material_frequency(df_train, df_test, feature='name_part1')
    except FileNotFoundError:
        print(f'train and test file not found in data/{feats_name}')
        sys.exit(1)

    if model_name == "lr_solidloading":
        print(model_name, "selected. Removing other features except Solid Loading")
        selected_feats = ['vf_total']

    model, search_space, selected_preprocessor, model_name = custom_models.get_model_by_name(
        seed, cat_cols, selected_feats, target, encode_min_frequency, model_name
    )

    # selected_preprocessor.fit(df_train[selected_feats], df_train[target])
    pipe = Pipeline([
        ('preprocess', selected_preprocessor),
        ('model', model)
    ])

    print("-"*40)
    print("Selected model:", model_name)
    print("Search space:", search_space)
    print("Fitting Grid...")
    start = time.time()
    if model_name == 'catb_native':
        cat_feature_indices = [df_train[selected_feats].columns.get_loc(col) for col in cat_cols]
        df_train[cat_cols] = df_train[cat_cols].fillna("None").astype(str)
        df_test[cat_cols] = df_test[cat_cols].fillna("None").astype(str)
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
    helpers.metrics_utils.get_test_metrics_by_group(df_test, test_preds, target, ['name_part1_freq_bin', 'year'])

    print("Test additional_data Data")
    additional_data_preds = best_clf.predict(df_additional_data[selected_feats])
    helpers.metrics_utils.get_regression_metrics(additional_data_preds, df_additional_data[target])
    additional_group_metrics = helpers.metrics_utils.get_test_metrics_by_group(df_additional_data, additional_data_preds, target, ['title', 'name_part1'])
    for key in additional_group_metrics.keys():
        additional_group_metrics[key] = pd.DataFrame(additional_group_metrics[key]).T
        print("Additional Data Group Metrics for", key)
        print(additional_group_metrics[key])
    df_additional_data['preds'] =  additional_data_preds
    df_additional_data['error'] = df_additional_data['preds'] - df_additional_data['porosity']
    print(df_additional_data[['title', 'year', 'doi', 'vf_total', 'name_part1','porosity', 'preds', 'error']].sort_values('year', ascending=False))
    print('paper mae')
    additional_group_metrics['recent_papers'] = df_additional_data[['title', 'year', 'doi', 'vf_total', 'name_part1','porosity', 'preds', 'error']].sort_values('year', ascending=False)

    r2_additional_data, mae_additional_data, mse_additional_data, mape_additional_data = helpers.metrics_utils.get_regression_metrics(additional_data_preds, df_additional_data[target])


    print("*"* 40)
    metrics_dict = helpers.metrics_utils.get_pipeline_metrics([best_clf], df_train, df_test, target, FEATURES_TO_ANALYZE, COLUMNS_NOT_NULL, selected_pipelines=[model_name])
    metrics_dict['additional_metrics'] = additional_group_metrics
    for key in metrics_dict['group_metrics'].keys():
        print(key, "-"*20, 'topn')
        print(metrics_dict['group_metrics'][key]['topn'])
        print(key, "-"*20, 'all')
        print(metrics_dict['group_metrics'][key]['all'])
        print('-' * 40)

    print("Training Results")
    r2_train, mae_train, mse_train, mape_train = helpers.metrics_utils.get_regression_metrics(train_preds, df_train[target])
    print(f"R²: {r2_train}, MAE: {mae_train}, MSE: {mse_train}, MAPE: {mape_train}")

    print("\nTest Results")
    r2, mae, mse, mape = helpers.metrics_utils.get_regression_metrics(test_preds, df_test[target])
    print(f"R²: {r2}, MAE: {mae}, MSE: {mse}, MAPE: {mape}")

    models_folder = f"pipe/{feats_name}"
    model_results_path, model_filename = utils.save_pipeline_reg(
        best_clf, seed, models_folder, model_name, selected_preprocessor.name,
        r2_train, r2, mae, mse
    )

    cols_p = np.char.replace(np.array(preprocessor.get_feature_names_out(), dtype=str), 'remainder__', '')
    # train_X_p = pd.DataFrame(preprocessor.transform(df_train[selected_feats]), columns=cols_p)
    # test_X_p = pd.DataFrame(preprocessor.transform(df_test[selected_feats]), columns=cols_p)
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
        shap_plotter.plot_dependence_for_features(shap_plotter.preprocessed_X.columns.to_list(), color_feature="Solid Loading")
        shap_plotter.plot_dependence_for_features(shap_plotter.preprocessed_X.columns.to_list(), color_feature="Temp Sinter")
        shap_plotter.plot_dependence_for_features(shap_plotter.preprocessed_X.columns.to_list(), color_feature="Disp. wf.")
        shap_plotter.plot_dependence_for_features(shap_plotter.preprocessed_X.columns.to_list(), color_feature="Temp. Sinter")
        shap_plotter.plot_dependence_for_features(shap_plotter.preprocessed_X.columns.to_list(), color_feature="Temp. Cold")
        shap_plotter.plot_dependence_for_features(shap_plotter.preprocessed_X.columns.to_list(), color_feature="Binder wf.")
        # shap_plotter.plot_dependence_for_features(shap_plotter.X.columns.to_list())

        # Single decision plot for feature importance
        # import matplotlib.pyplot as plt
        # import os
        # import shap
        # plt.close('all')
        # feature = 'wf_disp_1'
        # feature_idx = list(preprocessed_X.columns).index(feature)
        # save_path = f"{model_results_path}/shap/decision/{feature}.png"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # shap_vals = explainer(preprocessed_X)
        # min_idx = np.argmin(shap_values[:, feature_idx])
        # print(df_test.iloc[min_idx])
        # shap.decision_plot(explainer.expected_value, shap_values[min_idx], feature_names=preprocessed_X.columns.to_list(), link="logit")
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(save_path, bbox_inches="tight", dpi=300)
        # preprocessed_X
        # sh2 = explainer.shap_values(preprocessed_X2, check_additivity = False)



        # Find material has lowest solid loading importance



    print("Finished")

    return metrics_dict


if __name__ == "__main__":
    shap_opt=1
    """Running all possible pipeline trains by index"""
    model_names =  [
        # "catb_onehot[selected]",
        # "catb_native",
        "catb_onehot_impute",
        # "xgb_onehot", "xgb_impute",
        # "lr",
        #"lr_solidloading",
        #"rf"
                    ]
    for model_name in model_names:
        print(f"\n=== Training with model: {model_name} ===")
        train_model(model_name=model_name, feats_name="reduced_feats", shap_opt=shap_opt)
    # results = train_model()

