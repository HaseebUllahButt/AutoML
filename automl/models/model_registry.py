"""
Model registry - defines all available models and their hyperparameter spaces
"""

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier, DummyRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelRegistry:
    """Registry of all available models and their hyperparameter spaces"""
    
    @staticmethod
    def get_classification_models(fast_only=False):
        """Get classification models"""
        models = {
            'dummy': {
                'model': DummyClassifier(strategy='most_frequent'),
                'params': {},
                'fast': True,
            },
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                },
                'fast': True,
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                'fast': True,
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan'],
                },
                'fast': False,
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7],
                },
                'fast': True,
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                'fast': False,
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                },
                'fast': False,
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                },
                'fast': True,
            },
            'svm': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                },
                'fast': False,
            },
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                },
                'fast': False,
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = {
                'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 63, 127],
                },
                'fast': True,
            }
        
        if fast_only:
            models = {k: v for k, v in models.items() if v['fast']}
        
        return models
    
    @staticmethod
    def get_regression_models(fast_only=False):
        """Get regression models"""
        models = {
            'dummy': {
                'model': DummyRegressor(strategy='mean'),
                'params': {},
                'fast': True,
            },
            'linear_regression': {
                'model': LinearRegression(),
                'params': {},
                'fast': True,
            },
            'ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                },
                'fast': True,
            },
            'lasso': {
                'model': Lasso(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                },
                'fast': True,
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.2, 0.5, 0.8],
                },
                'fast': True,
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                'fast': True,
            },
            'knn': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan'],
                },
                'fast': False,
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                'fast': False,
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                },
                'fast': False,
            },
            'adaboost': {
                'model': AdaBoostRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                },
                'fast': True,
            },
            'svm': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                },
                'fast': False,
            },
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                },
                'fast': False,
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 63, 127],
                },
                'fast': True,
            }
        
        if fast_only:
            models = {k: v for k, v in models.items() if v['fast']}
        
        return models

    @staticmethod
    def get_model_info():
        """Get metadata about all available models for display"""
        return {
            'dummy': {
                'type': 'Baseline',
                'description': 'Simple baseline model (most frequent class / mean value)',
                'category': 'baseline',
            },
            'logistic_regression': {
                'type': 'Linear',
                'description': 'Linear logistic regression with L2 regularization',
                'category': 'linear',
            },
            'linear_regression': {
                'type': 'Linear',
                'description': 'Simple linear regression baseline',
                'category': 'linear',
            },
            'decision_tree': {
                'type': 'Tree',
                'description': 'Standalone decision tree classifier/regressor',
                'category': 'tree',
            },
            'knn': {
                'type': 'Distance',
                'description': 'K-Nearest Neighbors (slower on large datasets)',
                'category': 'distance',
            },
            'naive_bayes': {
                'type': 'Probabilistic',
                'description': 'Gaussian Naive Bayes classifier',
                'category': 'probabilistic',
            },
            'random_forest': {
                'type': 'Ensemble',
                'description': 'Ensemble of decision trees',
                'category': 'tree-ensemble',
            },
            'gradient_boosting': {
                'type': 'Ensemble',
                'description': 'Gradient Boosting (sequential ensemble)',
                'category': 'boosting',
            },
            'adaboost': {
                'type': 'Ensemble',
                'description': 'Adaptive Boosting (adaptive ensemble)',
                'category': 'boosting',
            },
            'svm': {
                'type': 'Kernel',
                'description': 'Support Vector Machine (slower on large datasets)',
                'category': 'kernel',
            },
            'xgboost': {
                'type': 'Ensemble',
                'description': 'Extreme Gradient Boosting (fast & powerful)',
                'category': 'boosting',
            },
            'lightgbm': {
                'type': 'Ensemble',
                'description': 'Light Gradient Boosting (fast & efficient)',
                'category': 'boosting',
            },
            'ridge': {
                'type': 'Linear',
                'description': 'Ridge Regression (L2 regularization)',
                'category': 'linear',
            },
            'lasso': {
                'type': 'Linear',
                'description': 'Lasso Regression (L1 regularization & feature selection)',
                'category': 'linear',
            },
            'elastic_net': {
                'type': 'Linear',
                'description': 'Elastic Net (combined L1+L2 regularization)',
                'category': 'linear',
            },
        }

    @staticmethod
    def is_baseline_model(model_name: str) -> bool:
        """Check if a model is a baseline (dummy) model"""
        return model_name == 'dummy'
