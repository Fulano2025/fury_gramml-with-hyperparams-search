COMPONENTS = {
    "Pipeline": {"class": "sklearn.pipeline.Pipeline"},
    "FeatureUnion": {"class": "sklearn.pipeline.FeatureUnion"},
    "AdaboostClassifier": {
        "class": "autosklearn.pipeline.components.classification.adaboost.AdaboostClassifier",
        "hyperparams": [
            {"name": "algorithm", "value": "SAMME.R", "values": ["SAMME", "SAMME.R"]},
            {"name": "learning_rate", "value": 0.1, "values": [0.01, 0.1, 0.5075]},
            {"name": "max_depth", "value": 1, "values": [1, 3, 6]},
            {"name": "n_estimators", "value": 50, "values": [50, 175, 300]},
        ],
    },
    "BernoulliNBClassifier": {
        "class": "autosklearn.pipeline.components.classification.bernoulli_nb.BernoulliNB",
        "hyperparams": [
            {"name": "alpha", "value": 1.0, "values": [0.01, 1.0, 25.0075]},
            {"name": "fit_prior", "value": "True", "values": ["False", "True"]},
        ],
    },
    "DecisionTreeClassifier": {
        "class": "autosklearn.pipeline.components.classification.decision_tree.DecisionTree",
        "hyperparams": [
            {"name": "criterion", "value": "gini", "values": ["entropy", "gini"]},
            {"name": "max_depth_factor", "value": 0.5, "values": [0.0, 0.5, 1.0]},
            {"name": "max_features", "value": 1.0, "values": [1.0]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
            {"name": "class_weight", "value": "None", "values": ["None", "balanced"]},
        ],
    },
    "ExtraTreesClassifier": {
        "class": "autosklearn.pipeline.components.classification.extra_trees.ExtraTreesClassifier",
        "hyperparams": [
            {"name": "bootstrap", "value": "False", "values": ["False", "True"]},
            {"name": "criterion", "value": "gini", "values": ["entropy", "gini"]},
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_features", "value": 0.5, "values": [0.0, 0.25, 0.5]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
        ],
    },
    "GaussianNBClassifier": {
        "class": "autosklearn.pipeline.components.classification.gaussian_nb.GaussianNB",
        "hyperparams": [],
    },
    "GradientBoostingClassifier": {
        "class": "autosklearn.pipeline.components.classification.gradient_boosting.GradientBoostingClassifier",
        "hyperparams": [
            {"name": "early_stop", "value": "off", "values": ["off", "train", "valid"]},
            {
                "name": "l2_regularization",
                "value": 1e-10,
                "values": [1e-10, 0.250000000075, 0.50000000005],
            },
            {"name": "learning_rate", "value": 0.1, "values": [0.01, 0.1, 0.2575]},
            {"name": "loss", "value": "auto", "values": ["auto"]},
            {"name": "max_bins", "value": 255, "values": [255]},
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_leaf_nodes", "value": 31, "values": [3, 31, 514]},
            {"name": "min_samples_leaf", "value": 20, "values": [1, 20, 50]},
            {"name": "scoring", "value": "loss", "values": ["loss"]},
            {"name": "tol", "value": 1e-07, "values": [1e-07]},
            {"name": "n_iter_no_change", "value": 10, "values": [1, 6, 10]},
            {"name": "validation_fraction", "value": 0.1, "values": [0.01, 0.1, 0.11]},
        ],
    },
    "KNearestNeighborsClassifier": {
        "class": "autosklearn.pipeline.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier",
        "hyperparams": [
            {"name": "n_neighbors", "value": 1, "values": [1, 25, 50]},
            {"name": "p", "value": 2, "values": [1, 2]},
            {"name": "weights", "value": "uniform", "values": ["distance", "uniform"]},
        ],
    },
    "LDAClassifier": {
        "class": "autosklearn.pipeline.components.classification.lda.LDA",
        "hyperparams": [
            {"name": "n_components", "value": 10, "values": [1, 10, 63]},
            {
                "name": "shrinkage",
                "value": "None",
                "values": ["None", "auto", "manual"],
            },
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
            {"name": "shrinkage_factor", "value": 0.5, "values": [0.0, 0.25, 0.5]},
        ],
    },
    "LibLinear_SVCClassifier": {
        "class": "autosklearn.pipeline.components.classification.liblinear_svc.LibLinear_SVC",
        "hyperparams": [
            {"name": "C", "value": 1.0, "values": [0.03125, 1.0, 8192.0234375]},
            {"name": "dual", "value": "False", "values": ["False"]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {"name": "intercept_scaling", "value": 1, "values": [1]},
            {
                "name": "loss",
                "value": "squared_hinge",
                "values": ["hinge", "squared_hinge"],
            },
            {"name": "multi_class", "value": "ovr", "values": ["ovr"]},
            {"name": "penalty", "value": "l2", "values": ["l1", "l2"]},
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
            {"name": "class_weight", "value": "None", "values": ["None", "balanced"]},
        ],
    },
    "LibSVM_SVCClassifier": {
        "class": "autosklearn.pipeline.components.classification.libsvm_svc.LibSVM_SVC",
        "hyperparams": [
            {"name": "C", "value": 1.0, "values": [0.03125, 1.0, 8192.0234375]},
            {
                "name": "gamma",
                "value": 0.1,
                "values": [3.0517578125e-05, 0.1, 2.0000228881835938],
            },
            {"name": "kernel", "value": "rbf", "values": ["poly", "rbf", "sigmoid"]},
            {"name": "max_iter", "value": -1, "values": [-1]},
            {"name": "shrinking", "value": "True", "values": ["False", "True"]},
            {
                "name": "tol",
                "value": 0.001,
                "values": [1e-05, 0.001, 0.025007500000000002],
            },
            {"name": "coef0", "value": 0.0, "values": [-1.0, -0.75, 0.0]},
            {"name": "degree", "value": 3, "values": [2, 3, 4]},
            {"name": "class_weight", "value": "None", "values": ["None", "balanced"]},
        ],
    },
    "MultinomialNBClassifier": {
        "class": "autosklearn.pipeline.components.classification.multinomial_nb.MultinomialNB",
        "hyperparams": [
            {"name": "alpha", "value": 1.0, "values": [0.01, 1.0, 25.0075]},
            {"name": "fit_prior", "value": "True", "values": ["False", "True"]},
        ],
    },
    "PassiveAggressiveClassifier": {
        "class": "autosklearn.pipeline.components.classification.passive_aggressive.PassiveAggressive",
        "hyperparams": [
            {"name": "C", "value": 1.0, "values": [1e-05, 1.0, 2.5000075]},
            {"name": "average", "value": "False", "values": ["False", "True"]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {"name": "loss", "value": "hinge", "values": ["hinge", "squared_hinge"]},
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
        ],
    },
    "QDAClassifier": {
        "class": "autosklearn.pipeline.components.classification.qda.QDA",
        "hyperparams": [
            {"name": "reg_param", "value": 0.0, "values": [0.0, 0.25, 0.5]}
        ],
    },
    "RandomForestClassifier": {
        "class": "autosklearn.pipeline.components.classification.random_forest.RandomForest",
        "hyperparams": [
            {"name": "bootstrap", "value": "True", "values": ["False", "True"]},
            {"name": "criterion", "value": "gini", "values": ["entropy", "gini"]},
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_features", "value": 0.5, "values": [0.0, 0.25, 0.5]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
        ],
    },
    "SGDClassifier": {
        "class": "autosklearn.pipeline.components.classification.sgd.SGD",
        "hyperparams": [
            {"name": "alpha", "value": 0.0001, "values": [1e-07, 0.0001, 0.025000075]},
            {"name": "average", "value": "False", "values": ["False", "True"]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {
                "name": "learning_rate",
                "value": "invscaling",
                "values": ["constant", "invscaling", "optimal"],
            },
            {
                "name": "loss",
                "value": "log",
                "values": ["hinge", "log", "modified_huber"],
            },
            {"name": "penalty", "value": "l2", "values": ["elasticnet", "l1", "l2"]},
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
            {
                "name": "epsilon",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
            {"name": "eta0", "value": 0.01, "values": [1e-07, 0.01, 0.025000075]},
            {"name": "l1_ratio", "value": 0.15, "values": [1e-09, 0.15, 0.25000000075]},
            {"name": "power_t", "value": 0.5, "values": [1e-05, 0.25001, 0.5]},
        ],
    },
    "AdaboostRegressor": {
        "class": "autosklearn.pipeline.components.regression.adaboost.AdaboostRegressor",
        "hyperparams": [
            {"name": "learning_rate", "value": 0.1, "values": [0.01, 0.1, 0.5075]},
            {
                "name": "loss",
                "value": "linear",
                "values": ["exponential", "linear", "square"],
            },
            {"name": "max_depth", "value": 1, "values": [1, 3, 6]},
            {"name": "n_estimators", "value": 50, "values": [50, 175, 300]},
        ],
    },
    "ARDRegressor": {
        "class": "autosklearn.pipeline.components.regression.ard_regression.ARDRegression",
        "hyperparams": [
            {"name": "alpha_1", "value": 1e-06, "values": [1e-10, 1e-06, 0.0002500001]},
            {
                "name": "alpha_2",
                "value": 1e-06,
                "values": [1e-10, 1e-06, 0.000250000075],
            },
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {
                "name": "lambda_1",
                "value": 1e-06,
                "values": [1e-10, 1e-06, 0.000250000075],
            },
            {
                "name": "lambda_2",
                "value": 1e-06,
                "values": [1e-10, 1e-06, 0.000250000075],
            },
            {"name": "n_iter", "value": 300, "values": [300]},
            {
                "name": "threshold_lambda",
                "value": 10000.0,
                "values": [1000.0, 10000.0, 25750.0],
            },
            {
                "name": "tol",
                "value": 0.001,
                "values": [1e-05, 0.001, 0.025007500000000002],
            },
        ],
    },
    "DecisionTreeRegressor": {
        "class": "autosklearn.pipeline.components.regression.decision_tree.DecisionTree",
        "hyperparams": [
            {
                "name": "criterion",
                "value": "mse",
                "values": ["friedman_mse", "mae", "mse"],
            },
            {"name": "max_depth_factor", "value": 0.5, "values": [0.0, 0.5, 1.0]},
            {"name": "max_features", "value": 1.0, "values": [1.0]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
        ],
    },
    "ExtraTreesRegressor": {
        "class": "autosklearn.pipeline.components.regression.extra_trees.ExtraTreesRegressor",
        "hyperparams": [
            {"name": "bootstrap", "value": "False", "values": ["False", "True"]},
            {
                "name": "criterion",
                "value": "mse",
                "values": ["friedman_mse", "mae", "mse"],
            },
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_features", "value": 1.0, "values": [0.1, 0.35, 0.6]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
        ],
    },
    "GaussianProcessRegressor": {
        "class": "autosklearn.pipeline.components.regression.gaussian_process.GaussianProcess",
        "hyperparams": [
            {
                "name": "alpha",
                "value": 1e-08,
                "values": [1e-14, 1e-08, 0.2500000000000075],
            },
            {
                "name": "thetaL",
                "value": 1e-06,
                "values": [1e-10, 1e-06, 0.000250000075],
            },
            {"name": "thetaU", "value": 100000.0, "values": [1.0, 25000.75, 50000.5]},
        ],
    },
    "GradientBoostingRegressor": {
        "class": "autosklearn.pipeline.components.regression.gradient_boosting.GradientBoosting",
        "hyperparams": [
            {"name": "early_stop", "value": "off", "values": ["off", "train", "valid"]},
            {
                "name": "l2_regularization",
                "value": 1e-10,
                "values": [1e-10, 0.250000000075, 0.50000000005],
            },
            {"name": "learning_rate", "value": 0.1, "values": [0.01, 0.1, 0.2575]},
            {"name": "loss", "value": "least_squares", "values": ["least_squares"]},
            {"name": "max_bins", "value": 255, "values": [255]},
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_iter", "value": 100, "values": [32, 100, 160]},
            {"name": "max_leaf_nodes", "value": 31, "values": [3, 31, 514]},
            {"name": "min_samples_leaf", "value": 20, "values": [1, 20, 50]},
            {"name": "scoring", "value": "loss", "values": ["loss"]},
            {"name": "tol", "value": 1e-07, "values": [1e-07]},
            {"name": "n_iter_no_change", "value": 10, "values": [1, 6, 10]},
            {"name": "validation_fraction", "value": 0.1, "values": [0.01, 0.1, 0.11]},
        ],
    },
    "KNearestNeighborsRegressor": {
        "class": "autosklearn.pipeline.components.regression.k_nearest_neighbors.KNearestNeighborsRegressor",
        "hyperparams": [
            {"name": "n_neighbors", "value": 1, "values": [1, 25, 50]},
            {"name": "p", "value": 2, "values": [1, 2]},
            {"name": "weights", "value": "uniform", "values": ["distance", "uniform"]},
        ],
    },
    "LibLinear_SVRRegressor": {
        "class": "autosklearn.pipeline.components.regression.liblinear_svr.LibLinear_SVR",
        "hyperparams": [
            {"name": "C", "value": 1.0, "values": [0.03125, 1.0, 8192.0234375]},
            {"name": "dual", "value": "False", "values": ["False"]},
            {"name": "epsilon", "value": 0.1, "values": [0.001, 0.1, 0.25075]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {"name": "intercept_scaling", "value": 1, "values": [1]},
            {
                "name": "loss",
                "value": "squared_epsilon_insensitive",
                "values": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            },
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
        ],
    },
    "LibSVM_SVRRegressor": {
        "class": "autosklearn.pipeline.components.regression.libsvm_svr.LibSVM_SVR",
        "hyperparams": [
            {"name": "C", "value": 1.0, "values": [0.03125, 1.0, 8192.0234375]},
            {"name": "epsilon", "value": 0.1, "values": [0.001, 0.1, 0.25075]},
            {"name": "kernel", "value": "rbf", "values": ["linear", "poly", "rbf"]},
            {"name": "max_iter", "value": -1, "values": [-1]},
            {"name": "shrinking", "value": "True", "values": ["False", "True"]},
            {
                "name": "tol",
                "value": 0.001,
                "values": [1e-05, 0.001, 0.025007500000000002],
            },
            {"name": "coef0", "value": 0.0, "values": [-1.0, -0.75, 0.0]},
            {"name": "degree", "value": 3, "values": [2, 3, 4]},
            {
                "name": "gamma",
                "value": 0.1,
                "values": [3.0517578125e-05, 0.1, 2.0000228881835938],
            },
        ],
    },
    "RandomForestRegressor": {
        "class": "autosklearn.pipeline.components.regression.random_forest.RandomForest",
        "hyperparams": [
            {"name": "bootstrap", "value": "True", "values": ["False", "True"]},
            {
                "name": "criterion",
                "value": "mse",
                "values": ["friedman_mse", "mae", "mse"],
            },
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_features", "value": 1.0, "values": [0.1, 0.35, 0.6]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
        ],
    },
    "RidgeRegressor": {
        "class": "autosklearn.pipeline.components.regression.ridge_regression.RidgeRegression",
        "hyperparams": [
            {"name": "alpha", "value": 1.0, "values": [1e-05, 1.0, 2.5000075]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {
                "name": "tol",
                "value": 0.001,
                "values": [1e-05, 0.001, 0.025007500000000002],
            },
        ],
    },
    "SGDRegressor": {
        "class": "autosklearn.pipeline.components.regression.sgd.SGD",
        "hyperparams": [
            {"name": "alpha", "value": 0.0001, "values": [1e-07, 0.0001, 0.025000075]},
            {"name": "average", "value": "False", "values": ["False", "True"]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {
                "name": "learning_rate",
                "value": "invscaling",
                "values": ["constant", "invscaling", "optimal"],
            },
            {
                "name": "loss",
                "value": "squared_loss",
                "values": ["epsilon_insensitive", "huber", "squared_loss"],
            },
            {"name": "penalty", "value": "l2", "values": ["elasticnet", "l1", "l2"]},
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
            {
                "name": "epsilon",
                "value": 0.1,
                "values": [1e-05, 0.025007500000000002, 0.05000500000000001],
            },
            {"name": "eta0", "value": 0.01, "values": [1e-07, 0.01, 0.025000075]},
            {"name": "l1_ratio", "value": 0.15, "values": [1e-09, 0.15, 0.25000000075]},
            {"name": "power_t", "value": 0.25, "values": [1e-05, 0.25, 0.25001]},
        ],
    },
    "Densifier": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.densifier.Densifier",
        "hyperparams": [],
    },
    "ExtraTreesPreprocessorClassification": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification.ExtraTreesPreprocessorClassification",
        "hyperparams": [
            {"name": "bootstrap", "value": "False", "values": ["False", "True"]},
            {"name": "criterion", "value": "gini", "values": ["entropy", "gini"]},
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_features", "value": 0.5, "values": [0.0, 0.25, 0.5]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_impurity_decrease", "value": 0.0, "values": [0.0]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
            {"name": "n_estimators", "value": 100, "values": [100]},
            {"name": "class_weight", "value": "None", "values": ["None", "balanced"]},
        ],
    },
    "ExtraTreesPreprocessorRegression": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression.ExtraTreesPreprocessorRegression",
        "hyperparams": [
            {"name": "bootstrap", "value": "False", "values": ["False", "True"]},
            {
                "name": "criterion",
                "value": "mse",
                "values": ["friedman_mse", "mae", "mse"],
            },
            {"name": "max_depth", "value": "None", "values": ["None"]},
            {"name": "max_features", "value": 1.0, "values": [0.1, 0.35, 0.6]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 0.0, "values": [0.0]},
            {"name": "n_estimators", "value": 100, "values": [100]},
        ],
    },
    "FastICA": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.fast_ica.FastICA",
        "hyperparams": [
            {
                "name": "algorithm",
                "value": "parallel",
                "values": ["deflation", "parallel"],
            },
            {"name": "fun", "value": "logcosh", "values": ["cube", "exp", "logcosh"]},
            {"name": "whiten", "value": "False", "values": ["False", "True"]},
            {"name": "n_components", "value": 100, "values": [10, 100, 510]},
        ],
    },
    "FeatureAgglomeration": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration.FeatureAgglomeration",
        "hyperparams": [
            {
                "name": "affinity",
                "value": "euclidean",
                "values": ["cosine", "euclidean", "manhattan"],
            },
            {
                "name": "linkage",
                "value": "ward",
                "values": ["average", "complete", "ward"],
            },
            {"name": "n_clusters", "value": 25, "values": [2, 25, 102]},
            {
                "name": "pooling_func",
                "value": "mean",
                "values": ["max", "mean", "median"],
            },
        ],
    },
    "KernelPCA": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.kernel_pca.KernelPCA",
        "hyperparams": [
            {"name": "kernel", "value": "rbf", "values": ["poly", "rbf", "sigmoid"]},
            {"name": "n_components", "value": 100, "values": [10, 100, 510]},
            {"name": "coef0", "value": 0.0, "values": [-1.0, -0.75, 0.0]},
            {"name": "degree", "value": 3, "values": [2, 3, 4]},
            {
                "name": "gamma",
                "value": 1.0,
                "values": [3.0517578125e-05, 1.0, 2.0000228881835938],
            },
        ],
    },
    "RandomKitchenSinks": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks.RandomKitchenSinks",
        "hyperparams": [
            {
                "name": "gamma",
                "value": 1.0,
                "values": [3.0517578125e-05, 1.0, 2.0000228881835938],
            },
            {"name": "n_components", "value": 100, "values": [50, 100, 2537]},
        ],
    },
    "LibLinear_Preprocessor": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor.LibLinear_Preprocessor",
        "hyperparams": [
            {"name": "C", "value": 1.0, "values": [0.03125, 1.0, 8192.0234375]},
            {"name": "dual", "value": "False", "values": ["False"]},
            {"name": "fit_intercept", "value": "True", "values": ["True"]},
            {"name": "intercept_scaling", "value": 1, "values": [1]},
            {
                "name": "loss",
                "value": "squared_hinge",
                "values": ["hinge", "squared_hinge"],
            },
            {"name": "multi_class", "value": "ovr", "values": ["ovr"]},
            {"name": "penalty", "value": "l1", "values": ["l1"]},
            {
                "name": "tol",
                "value": 0.0001,
                "values": [1e-05, 0.0001, 0.025007500000000002],
            },
            {"name": "class_weight", "value": "None", "values": ["None", "balanced"]},
        ],
    },
    "NoPreprocessing": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.no_preprocessing.NoPreprocessing",
        "hyperparams": [],
    },
    "Nystroem": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler.Nystroem",
        "hyperparams": [
            {"name": "kernel", "value": "rbf", "values": ["poly", "rbf", "sigmoid"]},
            {"name": "n_components", "value": 100, "values": [50, 100, 2537]},
            {"name": "coef0", "value": 0.0, "values": [-1.0, -0.75, 0.0]},
            {"name": "degree", "value": 3, "values": [2, 3, 4]},
            {
                "name": "gamma",
                "value": 0.1,
                "values": [3.0517578125e-05, 0.1, 2.0000228881835938],
            },
        ],
    },
    "PCA": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.pca.PCA",
        "hyperparams": [
            {
                "name": "keep_variance",
                "value": 0.9999,
                "values": [0.5, 0.7499750000000001, 0.99995],
            },
            {"name": "whiten", "value": "False", "values": ["False", "True"]},
        ],
    },
    "PolynomialFeatures": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.polynomial.PolynomialFeatures",
        "hyperparams": [
            {"name": "degree", "value": 2, "values": [2, 2, 3]},
            {"name": "include_bias", "value": "True", "values": ["False", "True"]},
            {"name": "interaction_only", "value": "False", "values": ["False", "True"]},
        ],
    },
    "RandomTreesEmbedding": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding.RandomTreesEmbedding",
        "hyperparams": [
            {"name": "bootstrap", "value": "True", "values": ["False", "True"]},
            {"name": "max_depth", "value": 5, "values": [2, 4, 5]},
            {"name": "max_leaf_nodes", "value": "None", "values": ["None"]},
            {"name": "min_samples_leaf", "value": 1, "values": [1, 6, 11]},
            {"name": "min_samples_split", "value": 2, "values": [2, 7, 12]},
            {"name": "min_weight_fraction_leaf", "value": 1.0, "values": [1.0]},
            {"name": "n_estimators", "value": 10, "values": [10, 35, 60]},
        ],
    },
    "SelectPercentileClassification": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification.SelectPercentileClassification",
        "hyperparams": [
            {"name": "percentile", "value": 50.0, "values": [1.0, 25.75, 50.0]},
            {
                "name": "score_func",
                "value": "chi2",
                "values": ["chi2", "f_classif", "mutual_info"],
            },
        ],
    },
    "SelectPercentileRegression": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression.SelectPercentileRegression",
        "hyperparams": [
            {"name": "percentile", "value": 50.0, "values": [1.0, 25.75, 50.0]},
            {
                "name": "score_func",
                "value": "f_regression",
                "values": ["f_regression", "mutual_info"],
            },
        ],
    },
    "SelectRates": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.select_rates.SelectRates",
        "hyperparams": [
            {"name": "alpha", "value": 0.1, "values": [0.01, 0.1, 0.135]},
            {"name": "mode", "value": "fpr", "values": ["fdr", "fpr", "fwe"]},
            {"name": "score_func", "value": "chi2", "values": ["chi2", "f_classif"]},
        ],
    },
    "TruncatedSVD": {
        "class": "autosklearn.pipeline.components.feature_preprocessing.truncatedSVD.TruncatedSVD",
        "hyperparams": [{"name": "target_dim", "value": 128, "values": [10, 74, 128]}],
    },
    "Balancing": {
        "class": "autosklearn.pipeline.components.data_preprocessing.balancing.balancing.Balancing",
        "hyperparams": [
            {"name": "strategy", "value": "none", "values": ["none", "weighting"]}
        ],
    },
    "NoEncoding": {
        "class": "autosklearn.pipeline.components.data_preprocessing.categorical_encoding.no_encoding.NoEncoding",
        "hyperparams": [],
    },
    "OneHotEncoder": {
        "class": "autosklearn.pipeline.components.data_preprocessing.categorical_encoding.one_hot_encoding.OneHotEncoder",
        "hyperparams": [],
    },
    "CategoryShift": {
        "class": "autosklearn.pipeline.components.data_preprocessing.category_shift.category_shift.CategoryShift",
        "hyperparams": [],
    },
    "CategoricalImputation": {
        "class": "autosklearn.pipeline.components.data_preprocessing.imputation.categorical_imputation.CategoricalImputation",
        "hyperparams": [],
    },
    "NumericalImputation": {
        "class": "autosklearn.pipeline.components.data_preprocessing.imputation.numerical_imputation.NumericalImputation",
        "hyperparams": [
            {
                "name": "strategy",
                "value": "mean",
                "values": ["mean", "median", "most_frequent"],
            }
        ],
    },
    "MinorityCoalescer": {
        "class": "autosklearn.pipeline.components.data_preprocessing.minority_coalescense.minority_coalescer.MinorityCoalescer",
        "hyperparams": [
            {
                "name": "minimum_fraction",
                "value": 0.01,
                "values": [0.0001, 0.01, 0.125075],
            }
        ],
    },
    "NoCoalescence": {
        "class": "autosklearn.pipeline.components.data_preprocessing.minority_coalescense.no_coalescense.NoCoalescence",
        "hyperparams": [],
    },
    "MinMaxScalerComponent": {
        "class": "autosklearn.pipeline.components.data_preprocessing.rescaling.minmax.MinMaxScalerComponent",
        "hyperparams": [],
    },
    "NoRescalingComponent": {
        "class": "autosklearn.pipeline.components.data_preprocessing.rescaling.none.NoRescalingComponent",
        "hyperparams": [],
    },
    "NormalizerComponent": {
        "class": "autosklearn.pipeline.components.data_preprocessing.rescaling.normalize.NormalizerComponent",
        "hyperparams": [],
    },
    "QuantileTransformerComponent": {
        "class": "autosklearn.pipeline.components.data_preprocessing.rescaling.quantile_transformer.QuantileTransformerComponent",
        "hyperparams": [
            {"name": "n_quantiles", "value": 1000, "values": [10, 510, 1000]},
            {
                "name": "output_distribution",
                "value": "uniform",
                "values": ["normal", "uniform"],
            },
        ],
    },
    "RobustScalerComponent": {
        "class": "autosklearn.pipeline.components.data_preprocessing.rescaling.robust_scaler.RobustScalerComponent",
        "hyperparams": [
            {"name": "q_max", "value": 0.75, "values": [0.7, 0.75, 0.9497499999999999]},
            {"name": "q_min", "value": 0.25, "values": [0.001, 0.076, 0.25]},
        ],
    },
    "StandardScalerComponent": {
        "class": "autosklearn.pipeline.components.data_preprocessing.rescaling.standardize.StandardScalerComponent",
        "hyperparams": [],
    },
    "VarianceThreshold": {
        "class": "autosklearn.pipeline.components.data_preprocessing.variance_threshold.variance_threshold.VarianceThreshold",
        "hyperparams": [],
    },
}
