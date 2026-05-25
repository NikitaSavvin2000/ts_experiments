# lstm_grid = {
#     "lstm_units": [32, 64],
#     "activation": ["swish"],
#     "recurrent_dropout_rate": [0.0],
#     "regularizers_l2": [1e-3],
#     "optimizer": ["adam"],
#     "batch_size": [32],
#     "epochs": [10, 20]
# }

lstm_grid = {
    "lstm_units": [32, 64],
    "activation": ["swish"],
    "recurrent_dropout_rate": [0.0],
    "regularizers_l2": [1e-3],
    "optimizer": ["adam"],
    "batch_size": [32],
    "epochs": [2, 3, 5]
}


xgb_grid = {
    "max_depth": [3, 6],
    "n_estimators": [500, 1000],
    "subsample": [0.85, 1.0],
    "colsample_bytree": [0.85, 1.0],
    "min_child_weight": [5, 7],
    "booster": ["gbtree"]
}

catboost_grid = {
    "learning_rate": [0.03, 0.05],
    "depth": [6, 8],
    "iterations": [500, 1000],
    "l2_leaf_reg": [3, 5],
    "subsample": [0.8, 1.0],
    "rsm": [0.8, 1.0],
    "bagging_temperature": [0, 1],
    "random_strength": [1],
    "loss_function": ["RMSE"],
    "boosting_type": ["Plain"]
}

lgbm_grid = {
    "learning_rate": [0.01, 0.05],
    "n_estimators": [500, 1000],
    "max_depth": [6, 10],
    "num_leaves": [31, 64],
    "subsample": [0.85, 1.0],
    "colsample_bytree": [0.85, 1.0],
    "min_child_samples": [20],
    "reg_lambda": [0.0, 1.0],
    "reg_alpha": [0.0, 0.1],
    "boosting_type": ["gbdt"]
}

linear_regression_grid = {
    "fit_intercept": [True, False],
    "positive": [False],
    "copy_X": [True],
    "n_jobs": [-1]
}

random_forest_grid = {
    "n_estimators": [300, 500],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1],
    "max_features": ["sqrt", 0.7],
    "bootstrap": [True]
}

svr_grid = {
    "kernel": ["rbf"],
    "C": [1.0, 10.0],
    "epsilon": [0.05, 0.1],
    "gamma": ["scale", 0.1],
    "shrinking": [True]
}

prophet_grid = {
    "changepoint_prior_scale": [0.01, 0.1],
    "seasonality_prior_scale": [0.1, 1.0],
    "seasonality_mode": ["additive", "multiplicative"],
    "changepoint_range": [0.8, 0.9],
    "n_changepoints": [10, 25]
}

patchtst_grid = {
    "input_size": [48, 96],
    "max_steps": [100, 200],
    "batch_size": [32],
    "learning_rate": [0.001, 0.0005],
    "hidden_size": [128],
    "n_heads": [4, 8],
    "dropout": [0.1],
    "patch_len": [4, 8],
    "stride": [4]
}

arima_grid = {
    "p": list(range(0, 8)),
    "d": [0, 1, 2],
    "q": list(range(0, 8)),
    "trend": ["n"]
}

sarima_grid = {
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2],
    "P": [0, 1],
    "D": [0, 1],
    "Q": [0, 1],
    "m": [7, 12],
    "trend": ["n"]
}

dlinear_grid = {
    "input_chunk_length": [48, 96],
    "output_chunk_length": [1, 3],
    "n_epochs": [10, 20],
    "batch_size": [32],
    "optimizer_lr": [1e-3, 1e-4],
    "hidden_size": [64],
    "kernel_size": [2, 3]
}

tcn_grid = {
    "filters": [32, 64],
    "kernel_size": [3, 5],
    "dilation_rates": [[1, 2, 4, 8]],
    "stacks": [1],
    "dropout": [0.1],
    "batch_size": [32],
    "epochs": [10, 20],
    "learning_rate": [1e-3, 3e-4],
    "clipnorm": [1.0],
    "use_layer_norm": [True]
}

transformer_grid = {
    "embed_dim": [32, 64],
    "num_heads": [4, 8],
    "ff_dim": [128],
    "dropout": [0.1],
    "dense_units": [16],
    "learning_rate": [5e-4, 1e-4],
    "batch_size": [64],
    "epochs": [10, 20]
}

models_grids = {
    "LSTM": lstm_grid,
    "XGBoost": xgb_grid,
    "CatBoost": catboost_grid,
    "LightGBM": lgbm_grid,
    "LinearRegression": linear_regression_grid,
    "RandomForest": random_forest_grid,
    "SVR": svr_grid,
    "Prophet": prophet_grid,
    "ARIMAX": arima_grid,
    "SARIMA": sarima_grid,
    "PatchTST": patchtst_grid,
    "DLinear": dlinear_grid,
    "TCN": tcn_grid,
    "Transformer": transformer_grid,
}


lstm_params_easy = {
    "lstm0_units": 16,
    "lstm1_units": 16,
    "lstm2_units": 8,
    "activation": "tanh",
    "recurrent_dropout_rate": 0.0,
    "regularizers_l2": 0.0,
    "optimizer": "adam",
    "batch_size": 16,
    "epochs": 5
}

xgb_params_easy = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 100,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 5,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "booster": "gbtree"
}

catboost_params_easy = {
    "learning_rate": 0.1,
    "depth": 4,
    "iterations": 200,
    "l2_leaf_reg": 3,
    "subsample": 1.0,
    "rsm": 1.0,
    "bagging_temperature": 0.0,
    "random_strength": 1,
    "loss_function": "RMSE",
    "boosting_type": "Plain"
}

lgbm_params_easy = {
    "learning_rate": 0.1,
    "n_estimators": 200,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_samples": 20,
    "reg_lambda": 0.0,
    "reg_alpha": 0.0,
    "boosting_type": "gbdt"
}

linear_regression_params_easy = {
    "fit_intercept": True,
    "positive": False,
    "copy_X": True,
    "n_jobs": -1
}

random_forest_params_easy = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True
}

svr_params_easy = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,
    "gamma": "scale",
    "shrinking": True
}

prophet_params_easy = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 1.0,
    "seasonality_mode": "additive",
    "changepoint_range": 0.8,
    "n_changepoints": 10
}

patchtst_params_easy = {
    "max_steps": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "hidden_size": 64,
    "n_heads": 4,
    "dropout": 0.1,
    "patch_len": 4,
    "stride": 4
}

arima_params_easy = {
    "p": 1,
    "d": 1,
    "q": 1,
    "trend": "n"
}

sarima_params_easy = {
    "p": 1,
    "d": 1,
    "q": 1,
    "P": 0,
    "D": 0,
    "Q": 0,
    "m": 12,
    "trend": "n"
}

dlinear_params_easy = {
    "output_chunk_length": 1,
    "n_epochs": 10,
    "batch_size": 16,
    "optimizer_lr": 1e-3,
    "hidden_size": 32,
    "kernel_size": 3
}

tcn_params_easy = {
    "filters": 16,
    "kernel_size": 3,
    "dilation_rates": [1, 2, 4],
    "stacks": 1,
    "dropout": 0.1,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 1e-3,
    "clipnorm": 1.0,
    "use_layer_norm": True
}

transformer_params_easy = {
    "embed_dim": 32,
    "num_heads": 4,
    "ff_dim": 64,
    "dropout": 0.1,
    "dense_units": 16,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 5
}

models_easy = {
    "LSTM": lstm_params_easy,
    "XGBoost": xgb_params_easy,
    "CatBoost": catboost_params_easy,
    "LightGBM": lgbm_params_easy,
    "LinearRegression": linear_regression_params_easy,
    "RandomForest": random_forest_params_easy,
    "SVR": svr_params_easy,
    "Prophet": prophet_params_easy,
    "ARIMAX": arima_params_easy,
    "SARIMA": sarima_params_easy,
    "PatchTST": patchtst_params_easy,
    "DLinear": dlinear_params_easy,
    "TCN": tcn_params_easy,
    "Transformer": transformer_params_easy
}