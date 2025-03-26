import time
import shap
import pandas as pd
from helpers import utils, shap_explainer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_parser import DataParser
import warnings
import numpy as np
import select_model
import sys
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 600)


def rename_columns_if_contains(columns, rename_dict):
    renamed_columns = []
    for col in columns:
        new_col_name = col
        # Replace dictionary keys with their corresponding values
        for key in rename_dict:
            if key in col:
                new_col_name = new_col_name.replace(key, rename_dict[key])
        # Remove '_sklearn'
        new_col_name = new_col_name.replace('_sklearn', '')
        # Replace '__' and '_' with spaces
        new_col_name = new_col_name.replace('__', ' ').replace('_', ' ')
        # Capitalize only the first letter of the final name
        new_col_name = new_col_name.capitalize()
        renamed_columns.append(new_col_name)
    return renamed_columns


"""General Setup"""
print("Parsing data...")
parser = DataParser()
target = parser.target

""" Opts """
seed = 6
cv = 5
k = 50  # number of featured to be reduced to with shap
# selected_preprocessor = Preprocessors.opd()

"""Fetch dataset"""
feats_name_input = input("\n Which features to use? [all, reduced, simplest]")
feats_name = feats_name_input + "_feats"
print("Loading Data...")
selected_feats = parser.feats_dict[feats_name]
selected_feats.remove(target)
encode_min_frequency = 0.2  # 0.002 for simplest. best 0.2 for catb

# remove features for testing
# selected_feats = [item for item in selected_feats if item not in ['technique', 'direction', 'name_disp_1']]
# selected_feats = [item for item in selected_feats if item not in ['name_disp_1']]

num_cols = [x for x in parser.all_num_cols if x in selected_feats and x != target]
cat_cols = [x for x in parser.all_cat_cols if x in selected_feats and x != target]
# Step 2: Identify ignored_feats
ignored_feats = [feat for feat in selected_feats if feat not in set(num_cols + cat_cols)]
try:
    df_train = pd.read_csv(f"data/{feats_name}/train.csv")
    df_test = pd.read_csv(f"data/{feats_name}/test.csv")
    models_folder = f"pipe/{feats_name}"
except FileNotFoundError:
    print(f'train and test file not found in data/{feats_name}_feats')
    sys.exit(1)
# if model_name == 'lr':
#     df_train, df_test = df_train.dropna(subset='vf_total'), df_test.dropna(subset='vf_total')
#     df_train, df_test = utils.handle_outliers(df_train), utils.handle_outliers(df_test)


# df_train = data_parser.rename_columns_df(df_train)
# df_test = data_parser.rename_columns_df(df_test)

print("selected_feats", df_train.columns)

"""Prepare models"""
model, search_space, selected_preprocessor, model_name = select_model.get_model_by_name(seed, cat_cols, selected_feats,
                                                                                        target, encode_min_frequency)
selected_preprocessor.fit(df_train[selected_feats], df_train[target])
prep_feats = selected_preprocessor.get_feature_names_out()
prep_feats_renamed = [name.replace('remainder__', 'num__') for name in prep_feats]

# tree_method="hist", device="cuda" for gpu. But raises warning in deployment
pipe = Pipeline([('preprocess', selected_preprocessor),
                 # ('selector', sel1),  # better just adjust min_frequency in one hot encode or drop manually
                 ('model', model
                  )])  # task_type="GPU", devices='0'
train_name = f"{selected_preprocessor.name}"

"""Grid search"""
print("Search space:", search_space)
print('Fitting Grid...')
start = time.time()
grid = GridSearchCV(estimator=pipe, param_grid=search_space, cv=cv, verbose=0,
                    scoring='r2')  # r2, neg_mean_absolute_error, explained_variance, neg_mean_squared_error
if model_name == 'catb2':
    cat_feature_indices = [df_train[selected_feats].columns.get_loc(col) for col in cat_cols]
    df_train[cat_cols] = df_train[cat_cols].fillna("None")
    df_test[cat_cols] = df_test[cat_cols].fillna("None")
    grid = grid.fit(df_train[selected_feats], df_train[target],
                        model__cat_features=cat_feature_indices)
else:
    grid = grid.fit(df_train[selected_feats], df_train[target])

 # Grid results
print("CV results")
print(grid.cv_results_)
# Print leaderboard
print("\nLeaderboard:")
cv_results = pd.DataFrame(grid.cv_results_)
cv_results = cv_results.sort_values(by="mean_test_score", ascending=False)
print(cv_results)

print("Best params \n", grid.best_params_)

# Save attributes in pipeline model
best_clf = grid.best_estimator_
best_clf.name = model_name
best_clf.cat_cols = cat_cols
best_clf.num_cols = num_cols
best_clf.feats_name = feats_name
best_clf.cv_results_ = grid.cv_results_
best_clf.selected_feats = selected_feats  # save selected cols inside model
best_clf.preprocessor_name = selected_preprocessor.name  # save preprocessor name inside model

estimator = grid.best_estimator_[-1]  # last step in pipeline
preprocessor = grid.best_estimator_[0]  # first step in pipeline
print("Elapsed", (time.time() - start) / 60, 'min')
train_preds = best_clf.predict(df_train[selected_feats])
test_preds = best_clf.predict(df_test[selected_feats])

print("Training Results")
r2_train, mae_train, mse_train, mape_train = utils.get_regression_metrics(train_preds, df_train[target])

print("\nTest Results")
r2, mae, mse, mape = utils.get_regression_metrics(test_preds, df_test[target])

# important_cols = ['']
# filtered_test_df = df_test[df_test[important_cols].notnull().all(axis=1)]
# print("\nTest Results no nulls", important_cols, format(len(filtered_test_df) / len(df_test), '.2f'),
#       "% of cases in test")
# test_X_nonulls = filtered_test_df.drop([target], axis=1)
# test_Y_nonulls = filtered_test_df[target]
# utils.get_metrics(best_clf, test_X_nonulls, test_Y_nonulls, opt_print=True)

"""Save Model - Locally"""
model_results_path, model_filename = utils.save_pipeline_reg(best_clf, seed, models_folder, model_name, train_name,
                                                             r2_train, r2, mae, mse)

"""Create preprocessed frames"""
cols_p = np.array([str(item) for item in preprocessor.get_feature_names_out()])
cols_p = np.char.replace(cols_p, 'remainder__', '')
train_X_p = pd.DataFrame(preprocessor.transform(df_train[selected_feats]), columns=cols_p)
test_X_p = pd.DataFrame(preprocessor.transform(df_test[selected_feats]), columns=cols_p)
train_X_p.columns, test_X_p.columns = cols_p, cols_p

print("Count of created preprocessed columns: ", len(cols_p))

run_shap = input("Run shap? [1 or 0]")
if int(run_shap) == 1:
    prevalence = df_train[target].mean()
    predicted_label = best_clf.predict(df_test)

    explainer, preprocessed_X, shap_values, shap_confusion_matrix_dict = (
        shap_explainer.get_shap_plotter_inputs(estimator, preprocessor, df_test,
                                               selected_feats, predicted_label, prevalence,
                                              df_test[parser.target]))

    shap_plotter = shap_explainer.ShapPlotter(
        explainer, preprocessed_X, shap_values,
        shap_confusion_matrix_dict,
        col_rename_dict=parser.col_rename_dict,
        save_dirpath=model_results_path + "/shap",
        dataframe_subdir_name="test",
        plots_title=f'{model_name}_test'
    )
    shap_explainer.save_explainer(explainer, shap_plotter.filepath_explainer)
    # TODO: add artifacts from shapley
    shap_plotter.run_all_plots()

# catb = parser.load_pipeline(rf"D:\MyGoogleDrive\PythonScripts2\FreezeCastingPredict\selected_models\all_feats\catb-1h-190640-6-r0.91-r0.82-ma0.059-ms0.008\model\catb-1h-190640-6-r0.91-r0.82-ma0.059-ms0.008.pickle")
# df_test2[['name_part2', 'name_mold_mat', 'name_bind1'] ] = np.nan
# catb_r2, catb_mae, catb_mse, catb_mape = utils.get_regression_metrics(catb_preds, df_test[target])