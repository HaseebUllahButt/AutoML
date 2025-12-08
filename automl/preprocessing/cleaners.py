"""
Data cleaning transformers
Handles missing values, outliers, duplicates, and data type conversions
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Dict, Any

from ..config.settings import AutoMLConfig


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Intelligently handles missing values based on column characteristics"""
    
    def __init__(self, config: AutoMLConfig = None, strategy='auto'):
        self.config = config or AutoMLConfig()
        self.strategy = strategy
        self.imputers_ = {}
        self.columns_to_drop_ = []
        
    def fit(self, X, y=None):
        """Learn imputation strategies"""
        X = X.copy()
        
        # Identify columns to drop (too much missing)
        missing_pct = X.isnull().sum() / len(X)
        self.columns_to_drop_ = missing_pct[missing_pct > self.config.MAX_MISSING_RATIO].index.tolist()
        
        # Fit imputers for remaining columns
        X_remaining = X.drop(columns=self.columns_to_drop_)
        
        for col in X_remaining.columns:
            if X_remaining[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_remaining[col]):
                    # For numeric: use median
                    imputer = SimpleImputer(strategy='median')
                else:
                    # For categorical: use most frequent
                    imputer = SimpleImputer(strategy='most_frequent')
                
                imputer.fit(X_remaining[[col]])
                self.imputers_[col] = imputer
        
        return self
    
    def transform(self, X):
        """Apply imputation"""
        X = X.copy()
        
        # Drop high-missing columns
        X = X.drop(columns=[col for col in self.columns_to_drop_ if col in X.columns])
        
        # Apply imputers
        for col, imputer in self.imputers_.items():
            if col in X.columns:
                transformed = imputer.transform(X[[col]])
                # SimpleImputer returns 2D array; flatten to Series to avoid pandas ValueError
                if hasattr(transformed, 'ndim') and transformed.ndim > 1:
                    transformed = transformed[:, 0]
                X[col] = transformed
        
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handles outliers using IQR or Z-score methods"""
    
    def __init__(self, config: AutoMLConfig = None, method='iqr'):
        self.config = config or AutoMLConfig()
        self.method = method
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        """Learn outlier bounds"""
        X = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            data = X[col].dropna()
            
            if self.method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.config.OUTLIER_IQR_MULTIPLIER * IQR
                upper = Q3 + self.config.OUTLIER_IQR_MULTIPLIER * IQR
            else:  # z-score
                mean = data.mean()
                std = data.std()
                lower = mean - self.config.OUTLIER_Z_THRESHOLD * std
                upper = mean + self.config.OUTLIER_Z_THRESHOLD * std
            
            self.bounds_[col] = {'lower': lower, 'upper': upper}
        
        return self
    
    def transform(self, X):
        """Cap outliers at learned bounds"""
        X = X.copy()
        
        for col, bounds in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        
        return X


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """Removes duplicate rows"""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.drop_duplicates()
        return X


class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    """Removes constant and low-variance features"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.constant_columns_ = []
        
    def fit(self, X, y=None):
        """Identify constant columns"""
        X = X.copy()
        
        for col in X.columns:
            if X[col].nunique() <= 1:
                self.constant_columns_.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                if X[col].std() < self.config.MIN_VARIANCE_THRESHOLD:
                    self.constant_columns_.append(col)
        
        return self
    
    def transform(self, X):
        """Remove constant columns"""
        X = X.copy()
        cols_to_drop = [col for col in self.constant_columns_ if col in X.columns]
        return X.drop(columns=cols_to_drop)


class HighCardinalityHandler(BaseEstimator, TransformerMixin):
    """Handles high cardinality categorical variables"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.high_cardinality_cols_ = []
        
    def fit(self, X, y=None):
        """Identify high cardinality columns"""
        X = X.copy()
        
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if X[col].nunique() > self.config.MAX_CATEGORIES_OHE:
                self.high_cardinality_cols_.append(col)
        
        return self
    
    def transform(self, X):
        """Drop or hash high cardinality columns"""
        X = X.copy()
        
        # For now, just drop them (can implement hashing later)
        # In production, we'd use feature hashing or target encoding
        cols_to_drop = [col for col in self.high_cardinality_cols_ if col in X.columns]
        
        # Check if these are likely ID columns
        for col in cols_to_drop.copy():
            cardinality_ratio = X[col].nunique() / len(X)
            if cardinality_ratio < 0.5:  # Not an ID, keep it
                cols_to_drop.remove(col)
        
        return X.drop(columns=cols_to_drop)


class DataTypeConverter(BaseEstimator, TransformerMixin):
    """Converts data types appropriately"""
    
    def __init__(self):
        self.dtypes_ = {}
        
    def fit(self, X, y=None):
        """Learn optimal data types"""
        X = X.copy()
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.dtypes_[col] = 'numeric'
            elif X[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(X[col], errors='raise')
                    self.dtypes_[col] = 'numeric'
                except:
                    self.dtypes_[col] = 'categorical'
            else:
                self.dtypes_[col] = str(X[col].dtype)
        
        return self
    
    def transform(self, X):
        """Apply data type conversions"""
        X = X.copy()
        
        for col, dtype in self.dtypes_.items():
            if col not in X.columns:
                continue
                
            if dtype == 'numeric':
                X[col] = pd.to_numeric(X[col], errors='coerce')
            elif dtype == 'categorical':
                X[col] = X[col].astype(str)
        
        return X


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    """Collapses rare categories into 'Other'"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.category_mappings_ = {}
        
    def fit(self, X, y=None):
        """Learn which categories are rare"""
        X = X.copy()
        
        for col in X.select_dtypes(include=['object', 'category']).columns:
            value_counts = X[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < self.config.RARE_CATEGORY_THRESHOLD].index.tolist()
            
            if rare_categories:
                self.category_mappings_[col] = rare_categories
        
        return self
    
    def transform(self, X):
        """Replace rare categories with 'Other'"""
        X = X.copy()
        
        for col, rare_cats in self.category_mappings_.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: 'Other' if x in rare_cats else x)
        
        return X
