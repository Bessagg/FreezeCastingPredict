from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold


class Preprocessors:
    """Saved preprocessor steps for Pipeline
        name functions as : NumStepName_NumStepName_CatStepName
    """

    @staticmethod
    def opd():
        name = 'opd'  # Only pandas
        preprocessor = ColumnTransformer(transformers=[], verbose_feature_names_out=False, remainder='passthrough')
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def onehot(cat_cols, encode_min_frequency):
        name = '1h'  # 1 hot encode
        """Define preprocessing steps"""
        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            # verbose_feature_names_out=False,  # removes remainder__ and cat__ prefix
            transformers=[
                ('cat',
                 OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency), cat_cols),
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def impute_missing(cat_cols):
        name = 'Imiss'
        """Define preprocessing steps"""

        # Define a pipeline for categorical columns
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            transformers=[
                ('cat', cat_transformer, cat_cols)
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def impute(num_cols):
        name = 'impute'
        """Define preprocessing steps"""
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean. Best mean
            # ('scaler', StandardScaler())  # Scale the numerical values
        ])

        # column transformer to combine the numerical and categorical pipelines
        # # min_frequency: if value is int categories with a smaller cardinality will be considered infrequent. If float categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent # handle_unkown: 'ignore' or , ‘infrequent_if_exist’

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
                                         transformers=[
                                             ('num', num_transformer, num_cols),
                                         ],
                                         remainder='passthrough'
                                         )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def pass_all():
        name = 'pass'
        """Define preprocessing steps"""

        # Define a pipeline for categorical columns
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            transformers=[
                ('cat', cat_transformer, [])
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def impute_1hot(impute_cols, cat_cols, encode_min_frequency):
        name = 'I1h'
        """Define preprocessing steps"""
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean. Best mean
            # ('scaler', StandardScaler())  # Scale the numerical values
        ])

        # column transformer to combine the numerical and categorical pipelines
        # # min_frequency: if value is int categories with a smaller cardinality will be considered infrequent. If float categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent # handle_unkown: 'ignore' or , ‘infrequent_if_exist’

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
                                         transformers=[
                                             ('num', num_transformer, impute_cols),
                                             ('cat',
                                              OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency,
                                                            ),
                                              cat_cols)
                                         ],
                                         remainder='passthrough'
                                         )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def impute_scale_1hot(num_cols, cat_cols, encode_min_frequency):
        name = 'IS1'
        """Define preprocessing steps"""
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean. Best mean
            ('scaler', StandardScaler()),  # Scale the numerical values
        ])

        # column transformer to combine the numerical and categorical pipelines
        # # min_frequency: if value is int categories with a smaller cardinality will be considered infrequent. If float categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent # handle_unkown: 'ignore' or , ‘infrequent_if_exist’

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency, ),
                 cat_cols)
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def impute_scale_binary_1hot(num_cols, cat_cols, encode_min_frequency, seed, n_bins=3):
        name = 'ISBi1h'
        """Define preprocessing steps"""
        """KBinsDiscretizer - Bin continuous data into intervals. n_binsint or array-like of shape (n_features,), 
        default=5 ‘onehot’: Encode the transformed result with one-hot encoding and return a sparse matrix. Ignored 
        features are always stacked to the right. ‘onehot-dense’: Encode the transformed result with one-hot encoding 
        and return a dense array. Ignored features are always stacked to the right. ‘ordinal’: Return the bin 
        identifier encoded as an integer value. strategy{‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’ 
        Strategy used to define the widths of the bins. ‘uniform’: All bins in each feature have identical widths. 
        ‘quantile’: All bins in each feature have the same number of points. ‘kmeans’: Values in each bin have the 
        same nearest center of a 1D k-means cluster"""
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean. Best mean
            ('scaler', StandardScaler()),
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
            ('discretizer', KBinsDiscretizer(n_bins=n_bins, encode='onehot', strategy='quantile', random_state=seed))
            # Scale the numerical values
        ])

        # column transformer to combine the numerical and categorical pipelines
        # # min_frequency: if value is int categories with a smaller cardinality will be considered infrequent. If float categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent # handle_unkown: 'ignore' or , ‘infrequent_if_exist’

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency, ),
                 cat_cols)
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def impute_scale_1hot_balance(num_cols, cat_cols, encode_min_frequency):
        name = 'IS1'
        """Define preprocessing steps"""
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean. Best mean
            ('scaler', StandardScaler()),  # Scale the numerical values
        ])

        # column transformer to combine the numerical and categorical pipelines # min_frequency: if value is int
        # categories with a smaller cardinality will be considered infrequent. If float categories with a smaller
        # cardinality than min_frequency * n_samples will be considered infrequent # handle_unkown: 'ignore' or ,
        # ‘infrequent_if_exist’

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency, ),
                 cat_cols)
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def tree_ft(num_cols, cat_cols, encode_min_frequency):
        name = 'IS1'
        """Define preprocessing steps"""
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean. Best mean
            ('scaler', StandardScaler()),  # Scale the numerical values
        ])

        # column transformer to combine the numerical and categorical pipelines
        # # min_frequency: if value is int categories with a smaller cardinality will be considered infrequent. If float categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent # handle_unkown: 'ignore' or , ‘infrequent_if_exist’

        preprocessor = ColumnTransformer(verbose_feature_names_out=False,
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency, ),
                 cat_cols),
            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor

    @staticmethod
    def onehot_var_ths(cat_cols, encode_min_frequency, ths=0.0):
        """Onehot with variance threshold"""
        # ths as 0, 0.1, 0.01, 0.001
        name = '1h_var'  # 1 hot encode
        """Define preprocessing steps"""
        cat_transformer = Pipeline([
            ('onehot',  OneHotEncoder(handle_unknown='ignore', min_frequency=encode_min_frequency, )),  # Impute missing values with mean. Best mean
            ('var_ths', VarianceThreshold(ths)),  # Scale the numerical values
        ])
        preprocessor = ColumnTransformer(
            verbose_feature_names_out=False,  # removes remainder__ and cat__ prefix
            transformers=[
                ('cat',
                 cat_transformer, cat_cols),

            ],
            remainder='passthrough'
        )
        preprocessor.name = name
        return preprocessor



# class DropColumnsByName(BaseEstimator, TransformerMixin):
#     def __init__(self, columns_to_drop):
#         self.columns_to_drop = columns_to_drop
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         # Drop columns by names
#         return X.drop(columns=self.columns_to_drop)
