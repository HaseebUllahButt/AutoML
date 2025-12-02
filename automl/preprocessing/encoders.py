"""
Feature encoding transformers
Handles categorical encoding, scaling, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, 
    MinMaxScaler, RobustScaler
)
from typing import List, Dict, Any


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Intelligently encodes categorical variables"""
    
    def __init__(self, max_categories=50):
        self.max_categories = max_categories
        self.encoders_ = {}
        self.encoded_columns_ = []
        
    def fit(self, X, y=None):
        """Learn encodings"""
        X = X.copy()
        
        for col in X.select_dtypes(include=['object', 'category']).columns:
            n_unique = X[col].nunique()
            
            if n_unique == 2:
                # Binary: use label encoding
                le = LabelEncoder()
                le.fit(X[col].fillna('missing'))
                self.encoders_[col] = ('label', le)
                self.encoded_columns_.append(col)
            
            elif n_unique <= self.max_categories:
                # Low cardinality: use one-hot encoding
                self.encoders_[col] = ('onehot', X[col].unique())
                self.encoded_columns_.append(col)
            
            else:
                # High cardinality: use label encoding (or target encoding in production)
                le = LabelEncoder()
                le.fit(X[col].fillna('missing'))
                self.encoders_[col] = ('label', le)
                self.encoded_columns_.append(col)
        
        return self
    
    def transform(self, X):
        """Apply encodings"""
        X = X.copy()
        
        for col, (method, encoder) in self.encoders_.items():
            if col not in X.columns:
                continue
            
            if method == 'label':
                # Handle unseen categories
                X[col] = X[col].fillna('missing')
                X[col] = X[col].apply(
                    lambda x: x if x in encoder.classes_ else 'missing'
                )
                X[col] = encoder.transform(X[col])
            
            elif method == 'onehot':
                # Create dummy variables
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                
                # Ensure all expected columns exist
                expected_cols = [f"{col}_{cat}" for cat in encoder[1:]]
                for exp_col in expected_cols:
                    if exp_col not in dummies.columns:
                        dummies[exp_col] = 0
                
                X = X.drop(columns=[col])
                X = pd.concat([X, dummies[expected_cols]], axis=1)
        
        return X


class NumericScaler(BaseEstimator, TransformerMixin):
    """Scales numeric features"""
    
    def __init__(self, method='standard'):
        self.method = method
        self.scaler_ = None
        self.numeric_cols_ = []
        
    def fit(self, X, y=None):
        """Learn scaling parameters"""
        X = X.copy()
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numeric_cols_) == 0:
            return self
        
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler_.fit(X[self.numeric_cols_])
        return self
    
    def transform(self, X):
        """Apply scaling"""
        X = X.copy()
        
        if self.scaler_ is not None and len(self.numeric_cols_) > 0:
            cols_to_scale = [col for col in self.numeric_cols_ if col in X.columns]
            if cols_to_scale:
                X[cols_to_scale] = self.scaler_.transform(X[cols_to_scale])
        
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates engineered features"""
    
    def __init__(self, create_interactions=False, create_polynomials=False):
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.numeric_cols_ = []
        
    def fit(self, X, y=None):
        """Identify numeric columns for feature engineering"""
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self
    
    def transform(self, X):
        """Create engineered features"""
        X = X.copy()
        
        if len(self.numeric_cols_) < 2:
            return X
        
        # Create ratios (simple feature engineering)
        if self.create_interactions and len(self.numeric_cols_) <= 10:
            for i, col1 in enumerate(self.numeric_cols_[:5]):  # Limit to prevent explosion
                for col2 in self.numeric_cols_[i+1:6]:
                    if col1 in X.columns and col2 in X.columns:
                        # Avoid division by zero
                        denominator = X[col2].replace(0, 1e-10)
                        X[f'{col1}_div_{col2}'] = X[col1] / denominator
        
        # Create polynomial features (squared terms)
        if self.create_polynomials and len(self.numeric_cols_) <= 20:
            for col in self.numeric_cols_[:10]:  # Limit to prevent explosion
                if col in X.columns:
                    X[f'{col}_squared'] = X[col] ** 2
        
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding for high-cardinality categoricals (for classification/regression)"""
    
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.target_means_ = {}
        self.global_mean_ = None
        self.categorical_cols_ = []
        
    def fit(self, X, y=None):
        """Learn target encodings"""
        if y is None:
            return self
        
        X = X.copy()
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.global_mean_ = y.mean()
        
        for col in self.categorical_cols_:
            # Compute mean target for each category with smoothing
            category_means = X.groupby(col)[y.name if hasattr(y, 'name') else 'target'].mean()
            category_counts = X.groupby(col).size()
            
            # Smoothing
            smoothed_means = (
                (category_counts * category_means + self.smoothing * self.global_mean_) /
                (category_counts + self.smoothing)
            )
            
            self.target_means_[col] = smoothed_means.to_dict()
        
        return self
    
    def transform(self, X):
        """Apply target encodings"""
        X = X.copy()
        
        for col in self.categorical_cols_:
            if col in X.columns:
                X[col] = X[col].map(self.target_means_[col]).fillna(self.global_mean_)
        
        return X
