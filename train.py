import time
import shap
import pandas as pd
from helpers import utils, plt_and_save_shap
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

"""General Setup"""
print("Parsing data...")
data_parser = DataParser()
target = data_parser.target

""" Opts """
seed = 6
cv = 5
k = 50  # number of featured to be reduced to with shap
# selected_preprocessor = Preprocessors.opd()

"""Fetch dataset"""
feats_name_input = input("\n Which features to use? [all, mid, reduced]")
feats_name = feats_name_input + "_cols"
print("Loading Data...")
try:
    df_train = pd.read_csv(f"data/{feats_name}/train.csv")
    df_test = pd.read_csv(f"data/{feats_name}/test.csv")
    models_folder = f"pipe/{feats_name}"
except FileNotFoundError:
    print(f'train and test file not found in data/{feats_name}_cols')
    sys.exit(1)
selected_cols = data_parser.feats_dict[feats_name]
selected_cols.remove(target)
encode_min_frequency = 0.2  #
num_cols = [x for x in data_parser.all_num_cols if x in selected_cols and x != target]
cat_cols = [x for x in data_parser.all_cat_cols if x in selected_cols and x != target]
# if model_name == 'lr':
#     df_train, df_test = df_train.dropna(subset='vf_total'), df_test.dropna(subset='vf_total')
#     df_train, df_test = utils.handle_outliers(df_train), utils.handle_outliers(df_test)


# df_train = data_parser.rename_columns_df(df_train)
# df_test = data_parser.rename_columns_df(df_test)

print("Selected_cols", df_train.columns)

"""Prepare models"""
model, search_space, selected_preprocessor, model_name = select_model.get_model_by_name(seed, cat_cols, selected_cols, target, encode_min_frequency)
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
grid = GridSearchCV(estimator=pipe, param_grid=search_space, cv=cv, verbose=0, scoring='r2')  # r2, neg_mean_absolute_error, explained_variance, neg_mean_squared_error
grid = grid.fit(df_train[selected_cols], df_train[target])
best_clf = grid.best_estimator_
best_clf.name = model_name
best_clf.feats_name = feats_name
best_clf.cv_results_ = grid.cv_results_
best_clf.selected_cols = selected_cols  # save selected cols inside model
best_clf.preprocessor_name = selected_preprocessor.name  # save preprocessor name inside model

estimator = grid.best_estimator_[-1]  # last step in pipeline
preprocessor = grid.best_estimator_[0]  # first step in pipeline
print("Elapsed", (time.time() - start) / 60, 'min')
train_preds = best_clf.predict(df_train[selected_cols])
test_preds = best_clf.predict(df_test[selected_cols])

print("CV results")
print(grid.cv_results_)
print("Best params \n", grid.best_params_)

print("Training Results")
r2_train, mae_train, mse_train = utils.get_regression_metrics(train_preds, df_train[target])

print("\nTest Results")
r2, mae, mse = utils.get_regression_metrics(test_preds, df_test[target])

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
train_X_p = pd.DataFrame(preprocessor.transform(df_train[selected_cols]), columns=cols_p)
test_X_p = pd.DataFrame(preprocessor.transform(df_test[selected_cols]), columns=cols_p)
train_X_p.columns, test_X_p.columns = cols_p, cols_p
print("Count of created preprocessed columns: ", len(cols_p))

"""Generate shap"""
print("Running Shap")
shap_start = time.time()
explainer = shap.TreeExplainer(estimator, feature_perturbation='interventional')
preds = best_clf.predict(df_test[selected_cols])
shap_dict = plt_and_save_shap.plt_and_save_shap(explainer, test_X_p, df_test[target], preds,
                                                model_results_path, model_name, train_name,
                                                'test', check_additivity=True)
print("Elapsed Shap:", (time.time() - shap_start) / 60, 'min')
shap_values, varimps_sorted_cols, varimps = shap_dict['shap_values'], shap_dict["varimps_sorted_cols"], shap_dict[
    "varimps"]
top_shap_indices = varimps.index[0:k]
top_shap_col_names = varimps.values[:k, 0]

#  ############################## Retrain with shapley feature selection

# """Retrain with only best features"""
# from sklearn.compose import ColumnTransformer
#
# pipe2 = Pipeline([
#     ('preprocessor', preprocessor),
#     ('column_selector', ColumnTransformer(verbose_feature_names_out=False, transformers=[
#         ('selected', 'passthrough', top_shap_indices)])),  # Keep only the selected top features
#     ('model', CatBoostRegressor(logging_level='Silent'))])
# grid_clf2 = GridSearchCV(pipe2, search_space, cv=cv, verbose=1, scoring='f1')
# grid_clf2 = grid_clf2.fit(df_train[selected_cols], df_train[target])
# best_clf2 = grid_clf2.best_estimator_
# best_clf2.name = model_name + "_k"
# best_clf2.preprocessor_name = best_clf.preprocessor_name
# pipe2.name = model_name + "_k"
# best_clf.selected_cols = selected_cols
# best_clf.selected_cols_k = varimps.values[0:k, 0]
# best_clf.preprocessor_name = selected_preprocessor.name  # save name from custom preprocessor
#
# estimator2 = grid_clf2.best_estimator_[-1]  # last step in pipeline
# preprocessor2 = grid_clf2.best_estimator_[0]  # first step in pipeline
# preprocessor22 = grid_clf2.best_estimator_[1]  # first step in pipeline
#
# print("CV results")
# print(grid_clf2.cv_results_)
#
# print("Training Results")
# auc2, precision2, recall2, f12 = pred_utils.get_metrics(best_clf2, train_X, train_y, opt_print=True)
#
# print("\nValidation Results")
# auc_v2, precision_v2, recall_v2, f1_v2 = pred_utils.get_metrics(best_clf2, valid_X, valid_y, opt_print=True)
#
# print("\nTest Results")
# auc_t2, precision_t2, recall_t2, f1_t2 = pred_utils.get_metrics(best_clf2, test_X, test_y, opt_print=True)
#
# """Save Model - Locally"""
# model_results_path2, model_filename2 = data_parser.save_pipeline(best_clf2, models_folder, model_name,
#                                                                  train_name + "_sh", auc_t2,
#                                                                  precision_t2, recall_t2, f12, f1_t2)
#
# """Create preprocessed frames"""
# # pipe2.named_steps['column_selector'].transformers_[0][1].columns = top_shap_col_names
# cols_p2 = np.array([str(item) for item in preprocessor2.get_feature_names_out()])
# train_X_p2 = pd.DataFrame(preprocessor2.transform(train_X), columns=cols_p2)
# train_X_p22 = pd.DataFrame(preprocessor22.transform(train_X_p2), columns=top_shap_col_names)  # filtered shap cols
# test_X_p2 = pd.DataFrame(preprocessor2.transform(test_X), columns=cols_p2)
# test_X_p22 = pd.DataFrame(preprocessor22.transform(test_X_p2), columns=top_shap_col_names)  # filtered shap cols
# # train_X_p.columns, test_X_p.columns = cols_p, cols_p
#
#
# """Generate shap"""
# print("Running Shap")
# shap_start = time.time()
# explainer2 = shap.TreeExplainer(estimator2, feature_perturbation='interventional')
# preds = best_clf2.predict(test_X)
# shap_dict = plt_and_save_shap.plt_and_save_shap(explainer2, test_X_p22, test_y, preds,
#                                                 model_results_path2, model_name, train_name,
#                                                 results_name='test', check_additivity=True)
# print("Elapsed Shap:", (time.time() - shap_start) / 60, 'min')
# shap_values2, varimps_sorted_cols2, varimps2 = shap_dict['shap_values'], shap_dict["varimps_sorted_cols"], shap_dict[
#     "varimps"]
# top_shap_indices = varimps.index[0:k]
# top_shap_col_names = varimps.values[:k, 0]
#
# #
# # """Generate CFA with dice_ml - must have inputter"""
# # import dice_ml
# # d = dice_ml.Data(dataframe=df_train, continuous_features=num_cols, outcome_name=target)
# # m = dice_ml.Model(model=best_clf, backend='sklearn', model_type='model')  # ['sklearn', 'TF1', 'TF2', 'PYT']
# # exp = dice_ml.Dice(d, m, method="genetic")  # method = genetic/kdtree/
# #
# #
# # query_instances = train_X[train_y == 1][-5:-1]  # passing x_train for those rows where y_train less than desired range
# # print(query_instances)
# # dice_exp_random = exp.generate_counterfactuals(query_instances, total_CFs=1, verbose=True)
