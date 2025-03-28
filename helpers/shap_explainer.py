import os
import matplotlib.pyplot as plt
import shap
import pandas as pd
import matplotlib
import pickle
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
import re
matplotlib.use('Agg')  # hide plots


# Function to clean and rename columns
def rename_columns(columns, rename_dict):
    new_columns = []
    for col in columns:
        # Remove _sklearn and any prefix before it (like infrequent_sklearn_)
        clean_col = re.sub(r'\b\w*_sklearn_', '', col)  # Remove prefix ending with "_sklearn_"
        clean_col = re.sub(r'_sklearn$', '', clean_col)  # Remove _sklearn at the end if present

        # Rename using dictionary, if present
        new_columns.append(rename_dict.get(clean_col, clean_col))

    return new_columns

def get_shap_confusion_matrix_dict(X, predicted_label, true_y, shap_values):
    X.reset_index(inplace=True, drop=True)
    true_y.reset_index(inplace=True, drop=True)
    FP = X[(predicted_label == 1) & (true_y == 0)]
    FN = X[(predicted_label == 0) & (true_y == 1)]
    VP = X[(predicted_label == 1) & (true_y == 1)]
    VN = X[(predicted_label == 0) & (true_y == 0)]
    shap_confusion_matrix_dict = {'FP': {'X': FP, 'shap_values': shap_values[FP.index]},
                                  'FN': {'X': FN, 'shap_values': shap_values[FN.index]},
                                  'VP': {'X': VP, 'shap_values': shap_values[VP.index]},
                                  'VN': {'X': VN, 'shap_values': shap_values[VN.index]}
                                 }
    return shap_confusion_matrix_dict

def get_shap_plotter_inputs(estimator, preprocessor, df, selected_cols, preds, prevalence, y_true=None):
    explainer = get_explainer_from_estimator(estimator)
    preprocessed_X = preprocess_and_cast(preprocessor, df, selected_cols)
    shap_values = get_shap_values(explainer, preprocessed_X)
    # Set confusion_matrix_dict as None if y_true is not passed
    shap_confusion_matrix_dict = (
        get_shap_confusion_matrix_dict(preprocessed_X, preds, y_true, shap_values)) if y_true is not None else None
    explainer.base_value = np.log(prevalence / (1 - prevalence))
    return explainer, preprocessed_X, shap_values, shap_confusion_matrix_dict


def preprocess_and_cast(preprocessor, df, selected_cols):
    """
    Apply preprocessing and ensure datatypes are retained from the original DataFrame.
    """
    # Transform the DataFrame using the preprocessor and create a DataFrame from the result
    transformed = preprocessor.transform(df[selected_cols])
    cols = preprocessor.get_feature_names_out()

    # Create DataFrame with transformed data and proper column names
    transformed_df = pd.DataFrame(transformed, columns=cols, index=df.index)

    # Cast common columns back to original datatypes
    for col in transformed_df.columns.intersection(df.columns):
        transformed_df[col] = transformed_df[col].astype(df[col].dtype)
    print("Preprocessed number of cols:", len(transformed_df.columns))
    return transformed_df


def refactor_x(X, model):
    """Refactors feature names and values for shapley plots"""

    def capitalize_words(item):
        return ' '.join([word.capitalize() for word in item.split('_')])

    def replace_starting_text(item, cat_columns, model):
        for cat_col in cat_columns:
            if cat_col.startswith("CID"):
                continue
            if item.startswith(cat_col):
                # Replace the starting text and capitalize each word appropriately
                out = item.replace(cat_col + "_", model.rename_dict[cat_col] + " ")
                out = capitalize_words(out)
                return out  # cat_col + "_", model.rename_dict[cat_col] + " "
        return item

    def process_words_arr(s):
        s_without_underscore = s.replace('_', ' ')
        processed_string = ' '.join(word.capitalize() for word in s_without_underscore.split())

        # Replace specific substrings
        processed_string = ((processed_string.replace('Cid', 'CID').replace('Sos', 'SOS')
                             .replace("Tgo", "TGO")).replace("Tgp", "TGP")
                            .replace("Pcr", "PCR").replace("Ps ", "PS "))
        # Remove CID group names: removes last text delimted by space
        if processed_string.startswith("CID "):
            parts = processed_string.rsplit(' ', 1)  # Split from the right once
            if len(parts) > 1:
                processed_string = parts[0]  # Keep the part before the last space
        if processed_string == "Código Classificacao":
            processed_string = "Código Classificação"
        return processed_string

    def rename_repeated_features(feature_list):
        feature_counts = {}
        renamed_features = []

        for feature in feature_list:
            if feature in feature_counts:
                feature_counts[feature] += 1
                new_feature = f"{feature}_{feature_counts[feature]}"
            else:
                feature_counts[feature] = 0
                new_feature = feature
            renamed_features.append(new_feature)

        return renamed_features

    feature_names = model.feat_names
    feature_names = [model.rename_dict.get(item, item) for item in feature_names]
    feature_names = [replace_starting_text(item, model.cat_cols, model) for item in feature_names]
    feature_names = np.vectorize(process_words_arr)(feature_names)
    feature_names = rename_repeated_features(feature_names)
    X.index = feature_names
    X_refactored = np.where(np.isnan(X.values), 'Nulo', X.values)  # Replace Values NaN by 'Nulo' in plot
    return X_refactored


def sort_by_importance(X, sh, max_features=None):
    # Sort X and sh by importance value
    if not max_features:
        max_features = len(X.columns)
    sorted_indices = np.argsort(-sh)  # abs?
    sorted_X = X[sorted_indices]
    sorted_sh = sh[sorted_indices]
    return sorted_X, sorted_sh


def get_explainer_from_estimator(estimator):
    return shap.TreeExplainer(estimator, feature_perturbation='interventional')


def save_explainer(explainer, filepath: str):
    pickle.dump(explainer, open(filepath, 'wb'))


def get_shap_values(explainer, preprocessed_X):
    return explainer.shap_values(preprocessed_X, check_additivity=False)


class ShapPlotter:
    """
    Loads shap variables.
    If explainer is None then use estimator

    """

    def __init__(self, explainer, preprocessed_X, shap_values,
                 shap_confusion_matrix_dict,
                 color_feature=None,
                 col_rename_dict=None,
                 save_dirpath=None, dataframe_subdir_name="", plots_title="",
                 check_additivity=False, is_regression=True):

        # base value formula: np.log(preds.mean() / (1 - preds.mean()))
        self.shap_start = time.time()
        self.is_regression = is_regression
        self.explainer = explainer
        self.shap_values = shap_values

        # Set paths as None if path_root is None
        self.filepath_explainer = f"{save_dirpath}/explainer.pkl" if save_dirpath else None
        self.dirpath_root = f"{save_dirpath}/{dataframe_subdir_name}" if save_dirpath else None

        self.plots_title = plots_title

        if not is_regression:
            self.shap_confusion_matrix_dict = shap_confusion_matrix_dict  # shap values by confusion matrix (VP, VN, FP, FN)
        else:
            self.shap_confusion_matrix_dict = {"X": preprocessed_X}

        # Get dataframe preprocessed by the pipeline (for example: one-hot-encoded/scaled dataframe)
        self.X = preprocessed_X  # preprocess_and_cast(preprocessor, df, selected_cols)
        if col_rename_dict:
            self.X.columns = rename_columns(self.X.columns, col_rename_dict)
        self.check_additivity = check_additivity
        self.cmap = 'RdBu'  # plasma, RdBu
        self.X_raw = self.X.copy()  # for debug
        self.n_all_features = len(self.X.columns)
        self.zoomed_display = 20  # number of variables to show in zoomed summary plot
        self.dirpath_single = self.dirpath_root + "/single"
        self.dirpath_dependence = self.dirpath_root + "/dependence"

        if not is_regression:
            self.folders = [self.dirpath_single, self.dirpath_dependence, self.dirpath_single + "/VP",
                            self.dirpath_single + "/VN", self.dirpath_single + "/FP", self.dirpath_single + "/FN"]
        else:
            self.folders = [self.dirpath_single, self.dirpath_dependence]
        # plot params
        self.plt_fmt = 'png'
        self.n_samples_multiple = 500  # samples for all multiple decision plots
        self.n_samples_single = 100  # samples for single decision plots
        self.max_features = 15  # number of columns in varimp to be selected for dependence plots
        self.heatmap_n = 200

        # output
        self.varimp_sorted_cols = None
        self.varimp_sorted_cols_indices = None

        if not os.path.isdir(self.dirpath_root):
            for folder in self.folders:
                os.makedirs(folder)

        # reset indexes
        self.X.reset_index(inplace=True, drop=True)
        self.feature_importance = self.get_feature_importance()
        self.feature_ordered_index = self.feature_importance.index.to_list()
        self.feature_ordered_names = self.feature_importance['col_name']


    def get_feature_importance(self):
        vals = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame(list(zip(self.X.columns, vals)),
                                          columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        return feature_importance

    def plot_varimp(self):
        plt.clf()
        shap.summary_plot(self.shap_values, plot_type="bar", features=self.X, feature_names=self.X.columns,
                          cmap=self.cmap)
        if self.dirpath_root:
            plt.savefig(f"{self.dirpath_root}/SHAP_varimp.{self.plt_fmt}", dpi=300)

    def plot_summary(self):
        plt.clf()
        shap.summary_plot(self.shap_values, features=self.X, feature_names=self.X.columns, plot_size=(18, 8),
                          max_display=self.X.shape[1])
        plt.title(f"{self.plots_title}")
        if self.dirpath_root:
            plt.savefig(f"{self.dirpath_root}/SHAP__Summary.{self.plt_fmt}", dpi=300)

    def plot_summary_zoomed(self):
        plt.clf()
        shap.summary_plot(self.shap_values, features=self.X, feature_names=self.X.columns, plot_size=(18, 8),
                          max_display=self.zoomed_display)
        plt.title(f"{self.plots_title}")
        if self.dirpath_root:
            plt.savefig(f"{self.dirpath_root}/SHAP__SummaryZoomed.{self.plt_fmt}", dpi=300)

    def plot_dependence(self):
        plt.clf()
        plt.rcParams.update({'font.size': 8})
        fig, axs = plt.subplots(nrows=len(self.feature_importance['col_name'][:self.max_features]),
                                ncols=self.max_features, figsize=(60, 30))
        for i, col in enumerate(self.feature_importance['col_name'][:self.max_features]):
            inds = shap.approximate_interactions(col, self.shap_values, self.X)
            for j in range(self.max_features):
                ind_name = self.X.columns[inds[j]]
                shap.dependence_plot(ind=col, interaction_index=inds[j], shap_values=self.shap_values, features=self.X,
                                     ax=axs[i, j])
                axs[i, j].set(xticklabels=[])
                axs[i, j].set(yticklabels=[])
                axs[i, j].tick_params(bottom=False)
        plt.tight_layout()
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
        if self.dirpath_root:
            plt.savefig(f"{self.dirpath_dependence}/Dependence_Plots.{self.plt_fmt}", bbox_inches='tight', dpi=300)

    def plot_heatmap(self):
        print("Plotting Shap Heatmap...")
        plt.clf()

        shap.plots.heatmap(self.explainer(self.X), max_display=self.max_features,
                           instance_order=self.explainer(self.X).sum(1), plot_width=30)
        if self.dirpath_root:
            plt.savefig(f"{self.dirpath_root}/Heatmap.{self.plt_fmt}", bbox_inches='tight', dpi=300)

    def plot_decision(self, T, sh, subdir=None, title=None, max_features=None):
        """
        Parameters:
            T (DataFrame): The dataset containing the features and its value.
            sh (numpy matrix): numpy matrix containing the shap_values.
            subdir (str): subdir name where plot will be saved.
            title (str): title of the plot.
            max_features (int): The number of features to display.
        """
        plt.clf()

        if max_features is None:
            max_features = len(T.columns)
        # if is a vector (single decision)
        if sh.ndim == 1:
            feature_order = np.argsort(sh)
        # if is a matrix (multi decision)
        else:
            column_means = np.mean(sh, axis=0)
            feature_order = np.argsort(column_means)

        shap.decision_plot(base_value=self.explainer.expected_value, shap_values=sh, features=T,
                           feature_names=T.columns.to_list(), ignore_warnings=True, link='logit',
                           new_base_value=self.explainer.base_value,
                           feature_order=feature_order,  # feature_order = sorted indices
                           # feature_display_range=range(feature_display, -1, -1)
                           feature_display_range=slice(None, -max_features - 1, -1)
                           )
        plt.title(title)

        if self.dirpath_root:
            dirpath = f"{self.dirpath_root}/{subdir}"
            os.makedirs(dirpath, exist_ok=True)
            plt.savefig(f"{dirpath}/{T.index[0]}.{self.plt_fmt}", bbox_inches='tight', dpi=300)

    def single_decision_plots(self, shap_confusion_matrix_dict, n_samples=100):
        # single decision plots
        for category in shap_confusion_matrix_dict.keys():
            # last n_samples
            T = shap_confusion_matrix_dict[category]['X'][-n_samples:]
            shi = shap_confusion_matrix_dict[category]['shap_values'][-n_samples:]
            for i in range(0, len(T) - 1):
                path_decision_subdir = f"SingleDecisionPlots/{category}" # if self.dirpath_root else None # not needed
                path_reduced_subdir = f"SingleDecisionPlots/reduced_{category}"#  if self.dirpath_root else None

                # single decision
                self.plot_decision(T[i:i + 1], shi[i], subdir=path_decision_subdir, title=category)
                # single decision reduced
                self.plot_decision(T[i:i + 1], shi[i], subdir=path_reduced_subdir, max_features=self.max_features, title=category)

    def multiple_decision_plots(self, shap_confusion_matrix_dict, n_samples=100):
        for category in shap_confusion_matrix_dict.keys():
            T = shap_confusion_matrix_dict[category]['X'][-n_samples:]
            shi = shap_confusion_matrix_dict[category]['shap_values'][-n_samples:]
            # multiple decisions
            self.plot_decision(T, shi, title=category, subdir="MultipleDecisionPlots")
            # self.plot_decision(T, shi, category=category, dirpath_root=dirpath_root,
            #                    prefix="reduced", feature_display=self.max_features)

    def binned_preds_plots(self, preds, true_y):

        return

    def plot_dependence_for_features(self, feature_list, color_feature=None):
        """
        Generate SHAP dependence plots for a given list of features.

        Parameters:
            feature_list (list): List of feature names to plot.
            color_feature (str): column name of feature that will be used as colorbar.
        """
        if not feature_list:
            print("Feature list is empty. Please provide feature names.")
            return

        plt.clf()
        for feature in feature_list:
            if feature not in self.X.columns:
                print(f"Feature '{feature}' not found in dataset. Skipping...")
                continue

            plt.figure(figsize=(8, 6))
            interaction_index = color_feature if color_feature in self.X.columns else None
            shap.dependence_plot(feature, self.shap_values, self.X, interaction_index=interaction_index)
            plt.title(f"SHAP Dependence Plot: {feature}")

            if self.dirpath_dependence:
                dirpath = f"{self.dirpath_dependence}/{color_feature}"
                if not os.path.isdir(dirpath):
                    os.makedirs(dirpath)
                save_path = f"{dirpath}/{feature}.{self.plt_fmt}"
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                print(f"Saved: {save_path}")
            else:
                plt.show()

    def run_all_plots(self, feature_list=None,  color_feature=None):
        self.plot_varimp()
        self.plot_summary()
        self.plot_summary_zoomed()
        if feature_list:
            self.plot_dependence_for_features(feature_list, color_feature)

        if not self.is_regression:
            self.single_decision_plots(self.shap_confusion_matrix_dict)
            self.multiple_decision_plots(self.shap_confusion_matrix_dict)
        # plot_heatmap()
        print("Shap Finished")

        # Add any additional plotting methods you want to run

# Example usage:
# plotter = ShapPlotter(explainer, X, true_y, preds, model_results_path, model_name, train_name, results_name)
# plotter.run_all_plots()
