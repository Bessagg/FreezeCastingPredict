import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import plotly.io as pio
import data_parser
import squarify
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test
from scipy.stats import kruskal
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
pio.renderers.default = "browser"
# Show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)


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

#%% Correlation heatmap Numerical only
plt.figure(figsize=(20, 12))

# Set the style to 'white' to remove grid and background
sns.set(style="white")

# Calculate the correlation matrix
df_num = df.select_dtypes(include=['number', 'float64'])
corr = df_num.corr()
corr = round(corr * 100, 0)

# Masking the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# Plotting the heatmap
heatmap = sns.heatmap(corr,
                      vmin=-100, vmax=100,
                      mask=mask,
                      annot=True,
                      cmap='BrBG',
                      fmt=".0f",  # Display values as integers
                      annot_kws={"fontsize": 20},
                      cbar_kws={'shrink': 0.8})  # Adjust colorbar size

# Remove grid and background
heatmap.grid(False)
# Increase colorbar fontsize
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=28)  # Increase colorbar fontsize
# Adjust tick labels' font size and rotation
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=28)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=28)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=38, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')

# Tight layout and show
plt.tight_layout()
plt.show()
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
# significance_threshold = 1e-4
filtered_p_values = {col: {groups: p for groups, p in group_p_values.items() if p > significance_threshold} for col, group_p_values in p_values.items()}

# Print the sorted, filtered p-values
for key in filtered_p_values.keys():
    vals = filtered_p_values[key]
    for key2 in vals.keys():
        val2 = vals[key2]
        i# f val2 >= significance_threshold:
            # print(key, key2, val2)


#%% Categorical Analysis
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

#%% Categorical data Porosity Distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, ks_1samp  # For One-way ANOVA (which will compare groups in pairs)

# Define statistical test function
stat_test = ttest_ind  # One-way ANOVA

# Set the name of the statistical test for display purposes
stat_test_name = "T-test"  # Default name for ANOVA (can be changed accordingly)

# List of categorical columns to use as the group
df_str = (df.select_dtypes(include=[object]))
categorical_columns = df_str.columns

# Initialize dictionary to store results
p_values_dict = {}

# Iterate over different group variables
for group_col in categorical_columns:
    # Get the top 3 most frequent groups in this categorical column
    top_groups = df[group_col].value_counts().index[:3]

    # Filter dataset to only include these top groups
    df_filtered = df[df[group_col].isin(top_groups)]

    # Prepare data for the chosen statistical test
    group_data = [df_filtered[df_filtered[group_col] == group]['Porosity'].dropna() for group in top_groups]
    group_data = [g for g in group_data if len(g) > 0]  # Remove empty groups

    # Compute p-values for all pairs of groups
    p_values = []
    for i in range(len(group_data)):
        for j in range(i + 1, len(group_data)):
            stat, p_value = stat_test(group_data[i], group_data[j])
            p_values.append((top_groups[i], top_groups[j], p_value))

    # Sort p-values to find the minimum p-value pair
    p_values_sorted = sorted(p_values, key=lambda x: x[2])

    # Get the pair with the smallest p-value
    category1, category2, min_p_value = p_values_sorted[0]

    # Save the result for this group
    p_values_dict[group_col] = (category1, category2, min_p_value)

    # Prepare the title for the plot with the smallest p-value
    title = f'{category1} vs {category2} | {stat_test_name} p-value: {min_p_value:.0e}'

    # Create FacetGrid plot using histogram (bin plot) and KDE
    g = sns.FacetGrid(df_filtered, row=group_col, height=1.6, aspect=4)
    g.map(sns.histplot, 'Porosity', bins=5, kde=True)  # Use a consistent bin count

    # Set the title with line breaks
    g.fig.suptitle(title, fontsize=16)

    # Adjust spacing to ensure title doesn't overlap
    g.fig.subplots_adjust(top=0.80)  # Increase space above plot for title
    plt.show()
    plt.savefig(f"images/Count of {group_col}.png")

# Print out the results for all categorical columns
for group_col, (cat1, cat2, p_value) in p_values_dict.items():
    print(f"For {group_col}, the pair with the smallest p-value is {cat1} vs {cat2} with p-value: {p_value:.2e}")
