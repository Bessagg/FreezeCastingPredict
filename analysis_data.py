# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import time
import pandas as pd
import numpy as np
import math as m
# MATPLOTLIB
import matplotlib.pyplot as plt
# matplotlib parameters for Latex
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import prince
import plotly.io as pio
import data_parser
import matplotlib
import squarify
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test
from scipy.stats import kruskal
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
pio.renderers.default = "browser"

# Load Data
parser = data_parser.DataParser()
df = pd.read_csv("data/all_feats/df_selected_feats.csv")  # dataframe for analysis
df = parser.rename_columns_df(df)
df_all = pd.read_csv(f'data/df_all_feats.csv')
pallete = "summer"

# Samples per paper
samples_per_paper = df_all.groupby('paper_ID').size()
samples_per_paper = pd.DataFrame(samples_per_paper.sort_values(ascending=False), columns=['values'])
top3_paper_ids = samples_per_paper.index[0:3]
top3_papers_doi = df_all[df_all['paper_ID'].isin(top3_paper_ids)].drop_duplicates(subset='paper_ID')[['paper_ID', 'doi']]
print("Distinct papers", df_all['paper_ID'].nunique())
# Plot the TreeMap
plt.figure(figsize=(10, 6))
ths = 2  # count ths
# large_rectangles = samples_per_paper[samples_per_paper >= ths]
samples_per_paper['label'] = ['' if value <= ths else value for value in samples_per_paper['values']]
colors = sns.color_palette(pallete, len(samples_per_paper['values']))  # Choose a colormap for the TreeMap
squarify.plot(sizes=samples_per_paper['values'], color=colors, alpha=0.7, label=samples_per_paper['label'])
plt.axis('off')
plt.show()
plt.savefig(f"images/samples_per_paper.png")


# Temp sinter
max_temp_sinter1_row = df_all[df_all['temp_sinter_1'] == df_all['temp_sinter_1'].max()]
doi_with_max_temp_sinter1 = max_temp_sinter1_row['doi'].values[0]
print("DOI with the maximum 'temp_sinter1':", doi_with_max_temp_sinter1)

# Null counts
print(df.head())
print(f"Count of Null values out of {len(df_all)} rows \n", df_all.isnull().sum())
nulls = round(df_all.isnull().mean() * 100, 2)
selected_nulls = nulls[parser.all_feats]
print(f"\nPercentage of Null Values:\n", round(df_all.isnull().mean() * 100, 2), "%")

# # Drop columns with less than min_n_rows as not null values
# for col in df.columns:
#     print(f'{col}_rows: {df[col].count()}')
#     if df[col].count() < min_n_rows:
#         df.drop(columns=col, axis=1, inplace=True)
#         print(f"*dropped {col}, less than {min_n_rows} rows")
#
# """technique and direction only have 610 rows and temp_cold 1643 :c """
# print(f'Selected columns with more than {min_n_rows}: \n{df.columns}')
#

# #################################### Correlation heatmap Numerical only
plt.figure(figsize=(18, 12))
df_num = df.select_dtypes(include=['number', 'float64'])
# plt.tight_layout()
plt.show()
plt.subplots_adjust(left=0.21, right=1.05, top=0.95, bottom=0.3)
corr = df_num.corr()

# null_percentages = df_num.isnull().mean() * 100
# columns_with_nulls_above_50 = null_percentages[null_percentages < 50].index.tolist()
# corr = df_num[columns_with_nulls_above_50].corr()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG', fmt=".2%", annot_kws={"fontsize": 20})
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=28)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=28)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=38, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')
# heatmap.set_title('Matriz de Correlação', fontdict={'fontsize': 18}, pad=12)
print("Correlation matrix \n")
plt.savefig(f"images/Correlation.png")
# df.corr()['porosity']

# Dis Plot Numerical
# for col in df_num:
#     sns.displot(df, x=col, hue='Group', multiple="stack", stat="density", common_norm=False)
#     plt.savefig(f"images/Num_dist_{col}.png")
#
#
plt.close('all')
# Dist Plot Numerical
order = df['Group'].value_counts().index.to_list()
df['group_order'] = df['Group'].astype(pd.CategoricalDtype(categories=order, ordered=True))
df_num_dist = df.sort_values(by='group_order')  # order raw df for numerical distribution plot
top4_group_order_categories = df_num_dist['group_order'].value_counts().head(3).index.to_list()
df_num_dist_top4 = df_num_dist[df_num_dist['group_order'].isin(top4_group_order_categories)]
for col in df_num:
    sns.set(font_scale=1.5)
    g = sns.FacetGrid(df_num_dist_top4, row='Group',
                      height=1.6, aspect=4)
    g.map(sns.kdeplot, col, bw_adjust=.6)
    g.set_ylabels('Density')
    plt.savefig(f"images/num_dist_{col}.png")

# Initialize a dictionary to store p-values
p_values = {col: {} for col in df_num}

# Calculate p-values for each pair of groups for each numerical column
for col in df_num:
    for i, group1 in enumerate(top4_group_order_categories):
        for group2 in top4_group_order_categories[i+1:]:
            group1_data = df_num_dist_top4[df_num_dist_top4['Group'] == group1][col]
            group2_data = df_num_dist_top4[df_num_dist_top4['Group'] == group2][col]
            p_value = ks_2samp(group1_data, group2_data).pvalue
            p_values[col][(group1, group2)] = p_value

# Filter p-values to only show those above 1e-4
significance_threshold = 0.05
significance_threshold = 1e-4
filtered_p_values = {col: {groups: p for groups, p in group_p_values.items() if p > significance_threshold} for col, group_p_values in p_values.items()}

# Print the sorted, filtered p-values
for key in filtered_p_values.keys():
    vals = filtered_p_values[key]
    for key2 in vals.keys():
        val2 = vals[key2]
        i# f val2 >= significance_threshold:
            # print(key, key2, val2)


# #################################### Categorical Analysis
# Plot porosidade against string columns
df_str = (df.select_dtypes(include=[object]))
# df_str = (df.select_dtypes(include=[object])).dropna()
count_filter_n = 50
rank_filter_n = 5
plt.close('all')
# Count of categorical data
for col in df_str.columns:
    f = plt.figure(figsize=(12, 8))
    f.set_figheight(12)
    plt.subplots_adjust(bottom=0.4)
    plt.suptitle(col, fontsize=48)
    top_n = 5
    top_samples = df.groupby(col)[col].count().sort_values(ascending=False)[0:top_n]
    # top_samples_columns = top_samples.axes[0].values
    # top_10_samples_columns = ['Al2O3', 'HAP', 'YSZ', 'Mullite', 'PZT', 'Bioglass', 'Si3N4', 'Al2O3/ZrO2', 'TiO2', 'SiO2']
    ax = top_samples.iloc[0:top_n].sort_values(ascending=False).plot(kind="bar", fontsize=38)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.bar_label(ax.containers[0], label_type='center', fontsize=38, color='black')
    ax.axes.get_yaxis().set_visible(False)
    ax.xaxis.set_label_text("")
    f.tight_layout()
    f.subplots_adjust(top=0.9)
    plt.savefig(f"images/Count of {col}.png")

    plt.show()
# plt.close("all")


# Categorical data Porosity Distribution
kruskal_pvals = []
for col in df_str.columns:
    sns.set(font_scale=1.5)
    filtered_df = df[df[col].notnull()]  # Remove null in column
    rank_filter_n = 3
    rank_filter = df_str[col].value_counts().head(rank_filter_n).axes[0]  # Filter top 5 in column
    count_filter = df.groupby(col).filter(lambda x: len(x) > count_filter_n)[col].unique()
    selected_filter = rank_filter  # change here for count or rank filtering
    filtered_df = filtered_df[filtered_df[col].isin(selected_filter)]
    df.groupby(col).count().sort_values(by=parser.target_renamed, ascending=False)

    # Perform Kruskal-Wallis test
    groups = [filtered_df[filtered_df[col] == val][parser.target_renamed] for val in selected_filter]
    kruskal_statistic, pval = kruskal(*groups)
    kruskal_pvals.append({'pval': pval, 'col': col})
    order = df.groupby(col).count().sort_values(by=parser.target_renamed, ascending=False).index[0:rank_filter_n]  # order by count
    filtered_df = filtered_df.sort_values(by=parser.target_renamed)
    g = sns.FacetGrid(filtered_df, row=col,
                      height=1.6, aspect=4, col_order=order)
    # g.map(sns.kdeplot, parser.target_renamed)
    g.map(sns.histplot, parser.target_renamed, kde=True)
    g.set_ylabels('Density')
    # Add annotation with p-value
    g.fig.suptitle(f'\n Kruskal p-value: {pval:.3e}', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.8)  # Adjust this value as needed to move the plots down and create space at the top
    print(f'Kruskal p-value: {pval:.3e} for top {rank_filter_n} categories in {col}')
    g.set(yticks=[], ylabel='Count')

    plt.savefig(f"images/Count Distribution of {col}.png", bbox_inches='tight')

    # rank_filter = df_str[col].value_counts().head(5)  # list to filter by rank

plt.show()
# plt.close("all")


# mca = prince.MCA()
# X = df_str.dropna()
# fig, ax = plt.subplots()
# mc = prince.MCA(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42).fit(X)
# mc.plot_coordinates(
#     X=X,
#     ax=None,
#     figsize=(6, 6),
#     show_row_points=True,
#     row_points_size=10,
#     show_row_labels=False,
#     show_column_points=True,
#     column_points_size=30,
#     show_column_labels=False,
#     legend_n_cols=1
# )
# print("MC eigen values", mc.eigenvalues_)

encoder = OneHotEncoder(handle_unknown='ignore')

# # #################################### Numerical Analysis

"""
Before you perform factor analysis, you need to evaluate the “factorability” of our dataset. 
Factorability means "can we found the factors in the dataset?". 
here are two methods to check the factorability or sampling adequacy:

Bartlett’s Test
Kaiser-Meyer-Olkin Test
"""
# num_cols = df.select_dtypes(include=[float]).columns
# df_num = df[num_cols].dropna()  #
# count_filter_n = 50

"""
Bartlett's test
In this Bartlett ’s test, the p-value is 0. 
The test was statistically significant, indicating that the observed correlation matrix is not an identity matrix.
"""

chi_square_value, p_value = calculate_bartlett_sphericity(df_num)
print(chi_square_value, p_value)

"""
Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis. 
It determines the adequacy for each observed variable and for the complete model. 
KMO estimates the proportion of variance among all the observed variable. 
Lower proportion id more suitable for factor analysis. KMO values range between 0 and 1. 
Value of KMO less than 0.6 is considered inadequate.
"""


# kmo_all, kmo_model = calculate_kmo(df_num)
# print(kmo_model)

# # ########### PCA Analysis Principal component Analysis
# n_components = 3
# pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])
# pca = PCA(n_components=n_components)
# # X = df_num[df_num.columns.drop('Porosidade')]
# X = df_num[df_num.columns]
# y = parser.target
#
# # components = pca.fit_transform(df_num)
# # components = pipeline.fit_transform(df_num)
# X_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)  # normalize data
# components = pca.fit_transform(X_scaled)
# # print(pd.DataFrame(pca.components_, columns=X_scaled.columns, index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5']))
# total_var = pca.explained_variance_ratio_.sum() * 100
# labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
# labels['color'] = parser.target
# fig = px.scatter_matrix(
#     components,
#     color=y,
#     dimensions=range(n_components),
#     labels=labels,
#     title=f'Total Explained Variance: {total_var:.2f}%',
# )
# fig.update_traces(diagonal_visible=False)
# fig.show()

# # 3D plot Variance
# fig = px.scatter_3d(
#     components, x=0, y=1, z=2, color=DataParser.target, title=f'Total Explained Variance: {total_var:.2f}%',
#     labels=labels
# )
# fig.show()
# exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
# px.area(x=range(1, exp_var_cumul.shape[0] + 1), y=exp_var_cumul,
#         labels={"x": "# Components", "y": "Explained Variance"})
# plt.close('all')
