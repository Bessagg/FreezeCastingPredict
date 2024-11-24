param_grid_catb = [
    {
        'model__n_estimators': [250],  # best:500 50, 100, 200
        'model__depth': [8],  # best:8
        # 'model__loss_function': ['Logloss'],
        'model__eval_metric': ['MAE'],  # mae mape, poisson, rmse
        # 'model__subsample': [0.9, 1],  # 0.5, 0.8 , 0.2
        # 'model__min_child_samples': [1],  # minimum number of training samples in a leaf
        # 'model__learning_rate': [0.01, 0.3],  # default=0.03
        # 'model__l2_leaf_reg': [0, 1, 2],  # 1, 3, 5
        # 'model__colsample_bylevel': [0.01, 0.05],  # 0.01, 0.05, 0.001 error on gpu
        'model__boosting_type': ["Plain"],  # "Ordered", "Plain"
        # 'model__logging_level': ['Silent'],
        # 'model__error_score': ['raise'],
        # 'model__bootstrap_type': ['Poisson']  # GPU
    }]

param_grid_catb_single = [
    {
        'model__n_estimators': [250],  # best:500 50, 100, 200
        'model__depth': [3],  # best:8
        # 'model__loss_function': ['Logloss'],
        'model__eval_metric': ['MAE'],  # mae mape, poisson, rmse
        # 'model__subsample': [0.9, 1],  # 0.5, 0.8 , 0.2
        # 'model__min_child_samples': [1],  # minimum number of training samples in a leaf
        # 'model__learning_rate': [0.01, 0.3],  # default=0.03
        # 'model__l2_leaf_reg': [0, 1, 2],  # 1, 3, 5
        # 'model__colsample_bylevel': [0.01, 0.05],  # 0.01, 0.05, 0.001 error on gpu
        # 'model__boosting_type':  ["Plain"],  # "Ordered", "Plain"
        # 'model__logging_level': ['Silent'],
        # 'model__error_score': ['raise'],
        # 'model__bootstrap_type': ['Poisson']  # GPU
    }]

param_grid_xgb = [
    {
        # 'model__eval_metric': ['aucpr'],
        'model__n_estimators': [100],  #
        # 'model__learning_rate': [0.2],
        # 'model__min_child_weight': [1, 2, 10, 50],
        # 'model__gamma': [2],
        # 'model__subsample': [1],
        # 'model__colsample_bytree': [0.5],
        # 'model__reg_alpha': [1],
        'model__max_depth': [5],  # best=5
        # 'model__tree_method': ['gpu_hist'],  # enables gpu
        # 'model__n_estimators': [2000],  # best 250
        # 'model__learning_rate': [0.2, 0.01, 0.001],
        # 'model__min_child_weight': [1, 10, 100, 1000],
        # 'model__gamma': [2, 20, 200, 2000],
        # 'model__subsample': [1, 0.5],  # best=1|
        # 'model__colsample_bytree': [1, 0.5],
        # 'model__reg_alpha': [0, 1, 20, 200],
        # 'model__reg_lambda': [0, 1, 20, 200],  # best=1
    }]

param_grid_xgb_single = [
    {
        'model__n_estimators': [100],  #
        # 'model__learning_rate': [0.2],
        # 'model__min_child_weight': [1, 2, 10, 50],
        # 'model__gamma': [2],
        # 'model__subsample': [1],
        # 'model__colsample_bytree': [0.5],
        # 'model__reg_alpha': [1],
        'model__max_depth': [5],  # best=5
        # 'model__tree_method': ['gpu_hist'],  # enables gpu
        # 'model__learning_rate': [0.2, 0.01, 0.001],
        # 'model__min_child_weight': [1, 10, 100, 1000],
        # 'model__gamma': [2, 20, 200, 2000],
        # 'model__subsample': [1, 0.5],  # best=1|
        # 'model__colsample_bytree': [1, 0.5],
        # 'model__reg_alpha': [0, 1, 20, 200],
        # 'model__reg_lambda': [0, 1, 20, 200],  # best=1
    }]

param_grid_lr = [{
    # 'model__linearregression_normalize': [True, False]
}]

param_grid_rf = [{
    'model__n_estimators': [50, 100, 200],
    'model__max_features': ['log2'],
    'model__max_depth': [10, 20, 40],
    # 'model__min_samples_split': [5],
    # 'model__min_samples_leaf': [5],
    # 'model__bootstrap': [False]
}]

# Define param_grid_nn for MLPRegressor (optional)
param_grid_nn = [
    {
        'model__hidden_layer_sizes': [(150, 50, 20)],
        'model__activation': ['tanh', 'relu', 'identity'],
        'model__solver': ['adam', 'lbfgs', 'sgd'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate': ['constant', 'adaptive'],
    }
]
