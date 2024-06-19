import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
import matplotlib
import pickle

# matplotlib.use('Agg')  # hide plots
# matplotlib.use('TkAgg')  # show plots


def plt_and_save_shap(explainer, X, true_y, preds, model_results_path, model_name, train_name, results_name,
                      check_additivity=False):
    """Generate shap results. Explainer should be built from train frame. X as dataframe"""
    """Results name = test/valid/train
    train_name:"""
    cmap = 'RdBu'  # plasma, RdBu
    X_raw = X.copy()  # for debug
    plt.close('all')
    n_cols_all = len(X.columns)
    singles_plots_n = 40
    zoomed_display = 30  # number of variables to show in zoomed summary plot
    dest_path = f"{model_results_path}/shap/{results_name}"
    dest_path_single = dest_path + "/single"
    dest_path_dependence = dest_path + "/dependence"
    dest_path_explainer = f"{model_results_path}/shap/" + "explainer.pickle"
    # feature_names = X.columns.values
    folders = [dest_path_single, dest_path_dependence, dest_path_single + "/TP", dest_path_single + "/TN", dest_path_single + "/FP", dest_path_single + "/FN"]
    if not os.path.isdir(dest_path):
        for folder in folders:
            os.makedirs(folder)
    # Reset indexes
    X.reset_index(inplace=True, drop=True)
    true_y.reset_index(inplace=True, drop=True)
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)
    print("shap_values created")
    new_base_value = np.log(preds.mean()/(1 - preds.mean()))
    plt_fmt = 'png'
    n_samples_all = 500
    n_samples = 100
    reduced_cols_n = 15  # number of columns in varimp to be selected for dependence plots
    n_plots_force = 20
    heatmap_n = 100

    # Save Explainer
    pickle.dump(explainer, open(dest_path_explainer, 'wb'))

    # Varimp name
    vals = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    importance_sorted_index = feature_importance.index.to_list()  # 'hclust' or index list
    varimps = feature_importance.head(reduced_cols_n)
    varimps_sorted_cols = varimps['col_name']

    # Varimp plot
    plt.clf()
    shap.summary_plot(shap_values, plot_type="bar", features=X, feature_names=X.columns,
                      cmap=cmap)  # Use column names
    plt.title(train_name + ' Varimp on training data')
    plt.savefig(f"{dest_path}/SHAP_varimp_{model_name}_{train_name}.{plt_fmt}")

    # Summary plot
    plt.clf()
    plt_shap = shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_size=(18, 8),
                                 max_display=X.shape[1])
    plt.title(f"train_name - {results_name}")
    plt.savefig(f"{dest_path}/SHAP__Summary_{model_name}_{train_name}.{plt_fmt}")

    # Summary plot Zoomed
    plt.clf()
    plt_shap = shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_size=(18, 8),
                                 max_display=int(zoomed_display))
    plt.title(train_name + f'{results_name}')
    plt.savefig(f"{dest_path}/SHAP__SummaryZoomed_{model_name}_{train_name}.{plt_fmt}")

    # Last n decision plot with logit

    # Results dataframe
    df_r_full = pd.DataFrame()
    df_r_full['true'] = true_y
    df_r_full['preds'] = preds
    # df_r_full['preds_p'] = preds_p
    df_r_full['miss'] = (df_r_full['true'] != df_r_full['preds']).astype(int)
    df_r_full['ind'] = df_r_full.index
    df_r = df_r_full.tail(n_samples)  # get only last n rows
    df_r.reset_index(inplace=True)
    n = -100
    n = -2  # Last index to include in multiple decision plots

    # Shap dict
    # Single decision plots

    if not X.empty:
        shap.decision_plot(base_value=explainer.expected_value, shap_values=shap_values, features=X,
                           feature_names=X.columns.values[0:n_cols_all], ignore_warnings=True, link='logit',
                           new_base_value=new_base_value, feature_order=importance_sorted_index,
                           feature_display_range=range(n_cols_all, -1, -1))
        plt.title(f"{n_samples} - {model_name}_{train_name}")
        plt.savefig(f"{dest_path}/SHAP_decision_{n}.{plt_fmt}", bbox_inches='tight')

    """Single sample plots"""
    for i in range(singles_plots_n):
        plt.clf()
        T = X[slice(-2 - i, -1 - i, 1)]  # get last samples
        sh = shap_values[slice(-2 - i, -1 - i, 1)]
        with open(f"{dest_path_single}/SHAP_Single_decision{i}.txt", "w") as f:  # Save preprocessed input as text
            print(T, file=f)

        # Single full decision plots
        # sh = explainer.shap_values(T, check_additivity=check_additivity)
        shap.decision_plot(base_value=explainer.expected_value, shap_values=sh, features=T,
                           feature_names=X.columns.values[0:n_cols_all], ignore_warnings=True,
                           link='logit', new_base_value=new_base_value,
                           # highlight=true_y[T.index],
                           feature_display_range=range(n_cols_all, -1, -1)
                           )  # logit changes model output to probability
        plt.title(f"{model_name}_{train_name}")
        plt.savefig(f"{dest_path_single}/SHAP_full_decision_{i}.{plt_fmt}", bbox_inches='tight')

        # Single reduced decision plot : only positive shapley values
        plt.clf()
        positive_indices = np.where(sh[0] > 0.11)[0]  # Get the indices of columns with positive values
        other_indices = np.where(sh[0] <= 0.11)[0]
        T['OUTRAS FEATURES'] = round(sh[0][other_indices].sum(), 1)
        sh2 = np.append(sh[0], T['OUTRAS FEATURES'])  # shapley values with other features column
        positive_indices = np.append(positive_indices, len(sh2) - 1)

        shap.decision_plot(base_value=explainer.expected_value, shap_values=sh2[positive_indices],
                           features=T.iloc[:, positive_indices],
                           feature_names=T.columns.values[positive_indices],
                           ignore_warnings=True,
                           link='logit', new_base_value=new_base_value,
                           # highlight=true_y[T.index],
                           feature_display_range=range(n_cols_all, -1, -1)
                           )  # logit changes model output to probability
        plt.title(f"Positive Impact variables for Patient - {model_name}_{train_name}")
        plt.savefig(f"{dest_path_single}/SHAP_reduced_decision_{i}.{plt_fmt}", bbox_inches='tight')

        # # Single Force plot
        # # plt.clf()
        # shap.getjs()
        # T = X[(preds == 1) & (true_y == 1)]  # TP
        # T = T[slice(-2 - i, -1 - i, 1)]
        # sh = explainer.shap_values(T, check_additivity=check_additivity)
        # force_cols = varimps_sorted_cols.index
        # shap.force_plot(base_value=explainer.expected_value, shap_values=sh, features=T,
        #                 feature_names=X.columns.values[0:reduced_cols_n],
        #                 matplotlib=True, link='logit')
        # plt.title(f"Pacient - {model_name}_{train_name}")
        # plt.savefig(f"{dest_path_single}/SHAP_Single_force_{i}_{reduced_cols_n}cols.{plt_fmt}", bbox_inches='tight')
        # plt.close('all')

    # # Single Waterfall plot - commented due to addivitiy error (happens in big trees)
    # shap.plots.waterfall(explainer(X)[T.index.values[0]], max_display=10)  # explain T value
    # plt.savefig(f"{dest_path_single}/Single_waterfall.{plt_fmt}", bbox_inches='tight')

    """Dependence plots"""
    n_plots_force = 15
    plt.close('all')
    plt.rcParams.update({'font.size': 8})  # Change the font size to 8
    fig, axs = plt.subplots(nrows=len(varimps_sorted_cols[:reduced_cols_n]), ncols=n_plots_force,
                            figsize=(60, 30))
    for i, col in enumerate(varimps_sorted_cols[:reduced_cols_n]):
        inds = shap.approximate_interactions(col, shap_values, X)  # index of high interactions with col
        for j in range(n_plots_force):
            ind_name = X.columns[inds[j]]
            shap.dependence_plot(ind=col, interaction_index=inds[j],
                                 shap_values=shap_values,
                                 features=X, ax=axs[i, j])
            # axs[i,j].set_title(f"{col} vs {ind_name}")
            axs[i,j].set(xticklabels=[])
            axs[i, j].set(yticklabels=[])
            axs[i, j].tick_params(bottom=False)
            # plt.savefig(
            #     f"{dest_path_dependence}/Dependence_{j}_{col.replace('/', '')}_{ind_name.replace('/', '')}cols.{plt_fmt}",
            #     bbox_inches='tight')
            # plt.close("all")
    plt.tight_layout()
    plt.savefig(f"{dest_path_dependence}/Dependence_Plots.{plt_fmt}", bbox_inches='tight')
    plt.close('all')

    # Heatmap
    # shap.plots.heatmap(explainer(X[-heatmap_n:-1]), max_display=30, instance_order=explainer(X[-heatmap_n:-1]).sum(1), plot_width=30)
    shap.plots.heatmap(explainer(X), max_display=40, instance_order=explainer(X).sum(1), plot_width=30)  # ordered by value
    plt.savefig(f"{dest_path}/Heatmap.{plt_fmt}", bbox_inches='tight')
    plt.close('all')
    print("Ended shapley plots")
    return {"shap_values": shap_values, "varimps_sorted_cols": varimps_sorted_cols, "varimps": feature_importance}


def simple_shap_plot(explainer, X, true_y, check_additivity):
    """Generate shap results. Explainer should be built from train frame."""
    plt.close('all')
    n_cols = len(X.columns)
    # Reset indexes
    cols = X.columns
    X.reset_index(inplace=True, drop=True)
    true_y.reset_index(inplace=True, drop=True)
    if len(X) > 10000:
        X = shap.sample(X, nsamples=10000)  # sampled shap values
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    else:
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)
    new_base_value = np.log(0.08 / (1 - 0.08))
    plt_fmts = ['png', 'svg']
    n_samples_all = 1000
    n_samples = 1000
    n_cols_force = 20  # 10
    max_dependences_plots = 20  # 10
    heatmap_n = 100  # number of samples for heatmap
    cmap = None

    # Varimp name
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    varimps = feature_importance.head(n_cols_force)
    varimps_sorted_cols = varimps['col_name']
    print("Varimps:\n", varimps_sorted_cols)

    # Summary plot
    plt_shap = shap.summary_plot(shap_values, features=X, feature_names=cols, plot_size=(18, 8),
                                 max_display=X.shape[1], cmap=cmap)

    # Summary plot Zoomed
    plt.plot()
    plt_shap = shap.summary_plot(shap_values, features=X, feature_names=cols, plot_size=(18, 8),
                                 max_display=int(X.shape[1] / 10), cmap=cmap)  # max_display=X.shape[1] / 4

    # Varimp plot
    plt.plot()
    shap.summary_plot(shap_values, plot_type="bar", features=X, feature_names=cols, cmap=cmap)  # Use column names
    plt.title('Varimp on training data')
    return {"shap_values": shap_values, "varimps_sorted_cols": varimps_sorted_cols, "varimps": feature_importance}
