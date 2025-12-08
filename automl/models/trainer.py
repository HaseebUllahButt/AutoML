"""
Model trainer with hyperparameter tuning and comprehensive error handling
Enhanced with validation, recovery, and detailed error tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import time
import warnings
from typing import Dict, List, Tuple, Any
import joblib
import logging

from ..config.settings import AutoMLConfig
from .model_registry import ModelRegistry
from ..utils.error_handlers import (
    TrainingException, ErrorContext, retry, InputValidator, TimeoutError
)


class ModelTrainer:
    """Trains multiple models and selects the best one"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.task_type = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.all_results = []
        self.warnings = []
        
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                     task_type: str = 'auto', fast_only: bool = False) -> Dict:
        """
        Train multiple models and return results with comprehensive validation
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification', 'regression', or 'auto'
            fast_only: Only train fast models
            
        Returns:
            Dictionary with training results
        
        Raises:
            TrainingException: If validation fails
        """
        self.warnings = []
        self.all_results = []
        
        try:
            # Validate inputs
            if X is None or y is None:
                raise TrainingException("X and y cannot be None", {'has_X': X is not None, 'has_y': y is not None})
            
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TrainingException(f"X must be DataFrame or ndarray, got {type(X).__name__}")
            
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise TrainingException(f"y must be Series or ndarray, got {type(y).__name__}")
            
            if len(X) != len(y):
                raise TrainingException(
                    f"X and y have different lengths: {len(X)} vs {len(y)}",
                    {'X_len': len(X), 'y_len': len(y)}
                )
            
            if len(X) == 0:
                raise TrainingException("X and y are empty", {'len': 0})
            
            # Determine task type
            if task_type == 'auto':
                self.task_type = self._infer_task_type(y)
            else:
                self.task_type = task_type
            
            self.warnings.append(f"INFO: Task type: {self.task_type}")
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = self._split_data(X, y)
            except Exception as e:
                raise TrainingException(f"Failed to split data: {str(e)[:200]}")
            
            # Get models
            registry = ModelRegistry()
            if self.task_type == 'classification':
                models = registry.get_classification_models(fast_only=fast_only)
            else:
                models = registry.get_regression_models(fast_only=fast_only)
            
            if not models:
                raise TrainingException(f"No models available for {self.task_type}")
            
            self.warnings.append(f"INFO: Training {len(models)} models")
            
            # Train each model with error recovery
            for model_name, model_config in models.items():
                self.warnings.append(f"INFO: Training {model_name}...")
                
                try:
                    result = self._train_single_model(
                        model_name, model_config,
                        X_train, X_test, y_train, y_test
                    )
                    self.all_results.append(result)
                except TimeoutError as e:
                    self.warnings.append(f"WARNING: {model_name} training timed out: {str(e)[:200]}")
                    self.all_results.append({
                        'model_name': model_name,
                        'error': f"Training timed out: {str(e)[:200]}",
                        'status': 'timeout',
                    })
                except Exception as e:
                    self.warnings.append(f"WARNING: {model_name} failed: {str(e)[:200]}")
                    self.all_results.append({
                        'model_name': model_name,
                        'error': str(e)[:200],
                        'status': 'failed',
                    })
            
            # Select best model
            self._select_best_model()
            
            return {
                'task_type': self.task_type,
                'best_model_name': self.best_model_name,
                'best_score': self.best_score,
                'all_results': self.all_results,
                'warnings': self.warnings,
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape,
            }
        
        except TrainingException:
            raise
        except Exception as e:
            raise TrainingException(
                f"Unexpected error during training: {str(e)[:200]}",
                {'error_type': type(e).__name__}
            )
    
    def _infer_task_type(self, y: pd.Series) -> str:
        """Infer whether task is classification or regression"""
        n_unique = y.nunique()
        
        if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
            return 'regression'
        else:
            return 'classification'
    
    def _split_data(self, X, y) -> Tuple:
        """Split data into train and test sets"""
        # Check for stratification (classification only)
        stratify = None
        if self.task_type == 'classification':
            # Check if we have enough samples per class
            value_counts = y.value_counts()
            min_samples = value_counts.min()
            
            if min_samples >= 2:
                stratify = y
            else:
                self.warnings.append(
                    f"WARNING: Cannot stratify - some classes have only {min_samples} sample(s)"
                )
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=stratify
            )
        except ValueError as e:
            # If stratification fails, try without it
            self.warnings.append(f"WARNING: Stratification failed, splitting without it: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=None
            )
        
        self.warnings.append(
            f"INFO: Split data - Train: {len(X_train)} rows, Test: {len(X_test)} rows"
        )
        
        return X_train, X_test, y_train, y_test
    
    def _train_single_model(self, model_name: str, model_config: Dict,
                            X_train, X_test, y_train, y_test) -> Dict:
        """Train a single model with hyperparameter tuning"""
        start_time = time.time()
        
        model = model_config['model']
        param_grid = model_config['params']
        
        # Hyperparameter tuning if params provided
        if param_grid:
            try:
                model = self._tune_hyperparameters(
                    model, param_grid, X_train, y_train
                )
            except Exception as e:
                self.warnings.append(
                    f"WARNING: Hyperparameter tuning failed for {model_name}: {str(e)[:100]}"
                )
                # Continue with default model
        
        # Train model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
        except Exception as e:
            raise Exception(f"Training failed: {e}")
        
        # Evaluate
        try:
            metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        except Exception as e:
            raise Exception(f"Evaluation failed: {e}")
        
        training_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics,
            'training_time': round(training_time, 2),
            'status': 'success',
        }
    
    def _tune_hyperparameters(self, model, param_grid, X_train, y_train):
        """Perform hyperparameter tuning"""
        # Determine CV strategy
        if self.task_type == 'classification':
            try:
                cv = StratifiedKFold(
                    n_splits=min(self.config.HP_SEARCH_CV, len(y_train)//2),
                    shuffle=True,
                    random_state=self.config.RANDOM_STATE
                )
            except ValueError:
                cv = self.config.HP_SEARCH_CV
        else:
            cv = KFold(
                n_splits=min(self.config.HP_SEARCH_CV, len(y_train)//2),
                shuffle=True,
                random_state=self.config.RANDOM_STATE
            )
        
        # Choose search method
        if self.config.HP_SEARCH_METHOD == 'grid':
            search = GridSearchCV(
                model, param_grid,
                cv=cv,
                n_jobs=-1,
                verbose=0
            )
        else:  # random
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=min(self.config.HP_SEARCH_ITERATIONS, len(param_grid) * 5),
                cv=cv,
                n_jobs=-1,
                random_state=self.config.RANDOM_STATE,
                verbose=0
            )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        
        return search.best_estimator_
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test) -> Dict:
        """Evaluate model on train and test sets"""
        metrics = {}
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics['train_accuracy'] = round(accuracy_score(y_train, y_train_pred), 4)
            metrics['test_accuracy'] = round(accuracy_score(y_test, y_test_pred), 4)
            
            try:
                # Multi-class might need different averaging
                metrics['train_f1'] = round(f1_score(
                    y_train, y_train_pred, average='weighted', zero_division=0
                ), 4)
                metrics['test_f1'] = round(f1_score(
                    y_test, y_test_pred, average='weighted', zero_division=0
                ), 4)
            except:
                metrics['train_f1'] = None
                metrics['test_f1'] = None
            
            try:
                metrics['train_precision'] = round(precision_score(
                    y_train, y_train_pred, average='weighted', zero_division=0
                ), 4)
                metrics['test_precision'] = round(precision_score(
                    y_test, y_test_pred, average='weighted', zero_division=0
                ), 4)
            except:
                metrics['train_precision'] = None
                metrics['test_precision'] = None
            
            try:
                metrics['train_recall'] = round(recall_score(
                    y_train, y_train_pred, average='weighted', zero_division=0
                ), 4)
                metrics['test_recall'] = round(recall_score(
                    y_test, y_test_pred, average='weighted', zero_division=0
                ), 4)
            except:
                metrics['train_recall'] = None
                metrics['test_recall'] = None
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_test_proba = model.predict_proba(X_test)[:, 1]
                        metrics['test_roc_auc'] = round(roc_auc_score(y_test, y_test_proba), 4)
                except:
                    metrics['test_roc_auc'] = None
            
        else:
            # Regression metrics
            metrics['train_rmse'] = round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4)
            metrics['test_rmse'] = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
            
            metrics['train_mae'] = round(mean_absolute_error(y_train, y_train_pred), 4)
            metrics['test_mae'] = round(mean_absolute_error(y_test, y_test_pred), 4)
            
            metrics['train_r2'] = round(r2_score(y_train, y_train_pred), 4)
            metrics['test_r2'] = round(r2_score(y_test, y_test_pred), 4)
        
        return metrics
    
    def _select_best_model(self):
        """Select the best model based on test performance"""
        successful_results = [r for r in self.all_results if r.get('status') == 'success']
        
        if not successful_results:
            self.warnings.append("ERROR: No models trained successfully!")
            return
        
        # Determine scoring metric
        if self.task_type == 'classification':
            metric_key = 'test_accuracy'
            best_result = max(successful_results, key=lambda x: x['metrics'].get(metric_key, 0))
        else:
            metric_key = 'test_r2'
            best_result = max(successful_results, key=lambda x: x['metrics'].get(metric_key, -np.inf))
        
        self.best_model = best_result['model']
        self.best_model_name = best_result['model_name']
        self.best_score = best_result['metrics'].get(metric_key)
        
        self.warnings.append(
            f"✓ Best model: {self.best_model_name} ({metric_key}={self.best_score})"
        )
    
    def predict(self, X):
        """Make predictions with best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        return self.best_model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model to save!")
        
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'task_type': self.task_type,
        }, filepath)
        
        self.warnings.append(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        data = joblib.load(filepath)
        self.best_model = data['model']
        self.best_model_name = data['model_name']
        self.task_type = data['task_type']
        
        self.warnings.append(f"✓ Model loaded from {filepath}")
