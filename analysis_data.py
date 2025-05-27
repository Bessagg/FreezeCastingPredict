import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import plotly.io as pio
import data_parser
import squarify
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, ks_1samp, spearmanr  # For One-way ANOVA (which will compare groups in pairs)
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test
from scipy.stats import kruskal
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
pio.renderers.default = "browser"
# Show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", 30)
dpi=200


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
plt.figure(figsize=(21, 9))
ths = 2  # count ths
# large_rectangles = samples_per_paper[samples_per_paper >= ths]
samples_per_paper['label'] = ['' if value <= ths else value for value in samples_per_paper['values']]
colors = sns.color_palette(pallete, len(samples_per_paper['values']))  # Choose a colormap for the TreeMap
plt.rcParams.update({'font.size': 18})
plt.yticks(fontsize=18)

squarify.plot(sizes=samples_per_paper['values'], color=colors, alpha=0.7, label=samples_per_paper['label'])
plt.axis('off')
plt.savefig(f"images/samples_per_paper.png", bbox_inches='tight', dpi=dpi)
plt.show()


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
zero_nan_percentage = ((df_all['cooling_rate'].isna() | (df_all['cooling_rate'] == 0)).mean()) * 100
print(f"Percentage of 0 or NaN in 'cooling_rate': {zero_nan_percentage:.2f}%")


#%% Correlation heatmap Numerical only

# Calculate the correlation matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define custom colormap (green, yellow, red with transparency)
n_bins = 100
cmap = LinearSegmentedColormap.from_list('green_yellow_red_transparent',
                                         ['#ff6b6b', '#ffffff',  '#aed476'], N=n_bins)
cmap = LinearSegmentedColormap.from_list('green_blue', [ "#00427d", '#ffffff', "#00652e"], N=n_bins)


# Calculate the correlation matrix
df_num = df.select_dtypes(include=['number', 'float64']).copy()
cols = ['Porosity'] + [col for col in df_num.columns if col != 'Porosity']
df_num = df_num[cols]
corr = df_num.corr()


# Optional: use Spearman's correlation
# corr, _ = spearmanr(df_num)  # Calculate Spearman correlation, the second value is p-values

corr = np.round(corr * 100, 0)  # Use np.round to round values to the nearest integer

# Masking the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# Plot
plt.figure(figsize=(21, 9))
# Set the style to 'white' to remove grid and background
sns.set(style="white")

# Plotting the heatmap with the custom colormap
heatmap = sns.heatmap(corr,
                      vmin=-100, vmax=100,
                      mask=mask,
                      annot=True,
                      cmap=cmap,  # Apply the custom colormap
                      fmt=".0f",  # Display values as integers
                      annot_kws={"fontsize": 28},
                      cbar_kws={'shrink': 0.8})  # Adjust colorbar size

# Format the annotations to include '%' sign
for text in heatmap.texts:
    text.set_text(f"{text.get_text()}%")  # Add '%' after each number

# Remove grid and background
heatmap.grid(False)

# Increase colorbar fontsize
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=36)  # Increase colorbar fontsize

# Adjust tick labels' font size and rotation
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=28)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=28)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=38, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')

# Tight layout and show
plt.tight_layout()
# Save the plot
plt.savefig(f"images/Correlation.png", dpi=dpi)
plt.show()




#%% Correlation heatmap of Missing Values
plt.figure(figsize=(21, 9))

# Set the style to 'white' to remove grid and background
sns.set(style="white")

# Calculate the missing values correlation matrix
df_missing = df.isnull()  # Create a boolean DataFrame of missing values
corr_missing = df_missing.corr()  # Correlation of missing values
corr_missing = round(corr_missing * 100, 0)  # Multiply by 100 to get percentage

# Masking the upper triangle
mask = np.triu(np.ones_like(corr_missing, dtype=bool), k=1)

# Plotting the heatmap for missing values correlation
heatmap = sns.heatmap(corr_missing,
                      vmin=-100, vmax=100,
                      mask=mask,
                      annot=True,
                      cmap=cmap, #'BrBG',
                      fmt=".0f",  # Display values as integers
                      annot_kws={"fontsize": 28},
                      cbar_kws={'shrink': 0.8})  # Adjust colorbar size

# Remove grid and background
heatmap.grid(False)

# Increase colorbar fontsize
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=36)  # Increase colorbar fontsize

# Adjust tick labels' font size and rotation
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=28)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=28)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=38, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')

# Tight layout and show
plt.tight_layout()
plt.show()
plt.savefig(f"images/Correlation_missing.png", dpi=dpi)

#%% Numerical Dist

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
    plt.savefig(f"images/num_dist_{col}.png", dpi=dpi)

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
# List of categorical columns to use as the group
df = pd.read_csv("data/all_feats/df_selected_feats.csv")  # dataframe for analysis
df = parser.rename_columns_df(df)
df_str = (df.select_dtypes(include=[object]))
categorical_columns = df_str.columns.to_list()
categorical_columns = ["Group", "Solid Name", "Fluid Name"]

for col in categorical_columns:
    # Get category counts
    category_counts = df[col].value_counts()

    # Identify the top 4 categories and group the rest as "Others"
    top_categories = category_counts.index[:5]
    df[col] = df[col].apply(lambda x: x if x in top_categories else 'Others')
    if col == 'Group':
        df[col] = df[col].replace({'Others': 'Pharmaceutical'})


    # Compute aggregated statistics
    summary = df.groupby(col)['Porosity'].agg(
        Count='count',
        Mean_Porosity=lambda x: round(x.mean()*100, 0),
        STD_Porosity=lambda x: round(x.std()*100, 0)
    )

    # Compute percentage of total and round
    summary['Total%'] = round(((summary['Count'] / summary['Count'].sum()) * 100)).round().astype(int)

    # Print the table
    print(f"\nSummary for {col}:")
    print(summary.sort_values(by='Count', ascending=False).to_string())

    # Create the boxplot
    plt.figure(figsize=(21, 9))
    sns.boxplot(x=col, y='Porosity', data=df, color="#00652e")

    # Format y-axis as percentage
    plt.gca().set_ylabel('Porosity (%)', fontsize=28)  # Set y-axis label with fontsize 28
    plt.gca().set_yticks([0, 0.25, 0.5, 0.75, 1])  # Define y-ticks
    plt.gca().set_yticklabels([f'{int(i * 100)}%' for i in [0, 0.25, 0.5, 0.75, 1]])  # Format as percentages

    # Set title and improve plot appearance
    # plt.title(f"Boxplot of Porosity by {col}", fontsize=28)
    plt.xticks(rotation=45, ha='right', fontsize=28)  # Rotate x labels for readability
    plt.yticks(fontsize=28)  # Set y-tick font size
    # plt.xlabel(col, fontsize=28)  # Set x-axis label
    plt.gca().set_xlabel('')  # Remove x-axis label
    plt.tight_layout()  # Ensure everything fits well within the plot
    plt.savefig(f"images/summary_of_{col}.png", dpi=dpi)
    # Display the plot
    plt.show()

#%%
# Plot porosidade against string columns

# df_str = (df.select_dtypes(include=[object])).dropna()
count_filter_n = 50
rank_filter_n = 5
plt.close('all')
# Count of categorical data
for col in df_str.columns:
    f = plt.figure(figsize=(21, 9))
    f.set_figheight(12)
    plt.subplots_adjust(bottom=0.4)
    # plt.suptitle(col, fontsize=48)
    top_n = 5
    top_samples = df.groupby(col)[col].count().sort_values(ascending=False)[0:top_n]
    ax = top_samples.iloc[0:top_n].sort_values(ascending=False).plot(kind="bar", fontsize=38, color="#00652e")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.bar_label(ax.containers[0], label_type='center', fontsize=38, color='#f0f0f0')
    ax.axes.get_yaxis().set_visible(False)
    ax.xaxis.set_label_text("")
    f.tight_layout()
    f.subplots_adjust(top=0.9)
    plt.savefig(f"images/Count of {col}.png", dpi=dpi)

    plt.show()
# plt.close("all")

#%% Categorical data Porosity Distribution
# Define statistical test function
stat_test = ttest_ind  # One-way ANOVA

# Set the name of the statistical test for display purposes
stat_test_name = "T-test"  # Default name for ANOVA (can be changed accordingly)

# Initialize dictionary to store results
p_values_dict = {}

# Iterate over different group variables
for group_col in categorical_columns:
    # Get the top 3 most frequent groups in this categorical column
    top_groups = df[group_col].value_counts().index[:3]

    # Filter dataset to only include these top groups
    df_filtered = df[df[group_col].isin(top_groups)]

    # Determine the minimum sample size among the top 3 groups
    min_sample_size = min(df_filtered[df_filtered[group_col] == group].shape[0] for group in top_groups)

    # Sample the same number of observations for each group
    balanced_group_data = [df_filtered[df_filtered[group_col] == group].sample(min_sample_size, random_state=42)['Porosity'].dropna()
                           for group in top_groups]

    # Compute p-values for all pairs of groups
    p_values = []
    for i in range(len(balanced_group_data)):
        for j in range(i + 1, len(balanced_group_data)):
            stat, p_value = stat_test(balanced_group_data[i], balanced_group_data[j])
            p_values.append((top_groups[i], top_groups[j], p_value))

    # Sort p-values to find the smallest one
    p_values_sorted = sorted(p_values, key=lambda x: x[2])

    # Get the pair with the smallest p-value
    category1, category2, min_p_value = p_values_sorted[0]

    # Save the result for this categorical feature
    p_values_dict[group_col] = (category1, category2, min_p_value)

    # Prepare the title for the plot
    title = f'{category1} vs {category2} | {stat_test_name} p-value: {min_p_value:.0e}'

    # Create FacetGrid plot using histogram (bin plot) and KDE
    g = sns.FacetGrid(df_filtered, row=group_col, height=1.6, aspect=4)
    g.map(sns.histplot, 'Porosity', bins=5, kde=True)  # Use a consistent bin count

    # Set the title with line breaks
    g.fig.suptitle(title, fontsize=28)

    # Adjust spacing to ensure title doesn't overlap
    g.fig.subplots_adjust(top=0.80)  # Increase space above plot for title
    plt.show()
    plt.savefig(f"images/Count of {group_col}.png", dpi=dpi)

# Print out the results for all categorical columns
for group_col, (cat1, cat2, p_value) in p_values_dict.items():
    print(f"For {group_col}, the pair with the smallest p-value is {cat1} vs {cat2} with p-value: {p_value:.2e}")
