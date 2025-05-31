# Features for analysis
import pandas as pd


def add_material_novelty_feature(df_train: pd.DataFrame, df_test: pd.DataFrame, feature='name_part1', min_count=6):
    """
    Adds a boolean column '{feature}_novel_in_test' to df_test and df_train indicating if
    the feature value appears less than min_count times in df_train.
    """
    # Count occurrences of each feature value in train
    train_counts = df_train[feature].value_counts()

    # Materials in test with count < min_count in train
    def is_novel(val):
        return train_counts.get(val, 0) < min_count

    # Add column to test
    df_test[f'{feature}_novel_in_test'] = df_test[feature].apply(is_novel)

    # Optionally add the same column to train (False since count >= min_count or present in train)
    df_train[f'{feature}_novel_in_test'] = df_train[feature].apply(lambda v: train_counts.get(v, 0) < min_count)
    print("Material in test but has less than", min_count, "occurrences in train:")
    novel_materials = df_test.loc[df_test['name_part1_novel_in_test'], 'name_part1']
    print(f"Unique materials in test with less than {min_count} occurrences in train:")
    print(novel_materials.value_counts())
    return df_train, df_test


def add_bin_material_frequency(df_train: pd.DataFrame, df_test: pd.DataFrame, feature='name_part1'):
    """
    Adds a new feature to both df_train and df_test indicating the frequency bin
    of each material based on its count in df_train.
    """
    # Count occurrences in df_train
    material_counts = df_train[feature].value_counts()

    # Define binning function
    def assign_bin(count):
        if count <= 5:
            return '5'
        if count <= 10:
            return '10'
        elif count <= 50:
            return '50'
        elif count <= 200:
            return '200'
        else:
            return '200+'

    # Map counts to bins
    material_bins = material_counts.apply(assign_bin)

    # Map bins to df_train and df_test
    df_train[f'{feature}_freq_bin'] = df_train[feature].map(material_bins).fillna('Rare')
    df_test[f'{feature}_freq_bin'] = df_test[feature].map(material_bins).fillna('Rare')
    return df_train, df_test
