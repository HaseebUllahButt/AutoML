"""
Preprocessing pipeline builder
Orchestrates all preprocessing steps based on data profiling
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Tuple, Any

from ..config.settings import AutoMLConfig
from ..data.profiling import DataProfiler
from ..data.validation import ColumnValidator
from .cleaners import (
    MissingValueHandler, OutlierHandler, DuplicateRemover,
    ConstantFeatureRemover, HighCardinalityHandler,
    DataTypeConverter, RareCategoryCollapser
)
from .encoders import CategoricalEncoder, NumericScaler, FeatureEngineer


class PreprocessingPipelineBuilder:
    """Builds a complete preprocessing pipeline based on data profile"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.pipeline = None
        self.preprocessing_steps = []
        self.warnings = []
        self.target_column = None
        
    def build_pipeline(self, df: pd.DataFrame, target_col: str, 
                       profile: Dict = None) -> Pipeline:
        """
        Build a complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            profile: Data profile from DataProfiler (optional)
            
        Returns:
            Fitted sklearn Pipeline
        """
        self.target_column = target_col
        self.warnings = []
        self.preprocessing_steps = []
        
        # Generate profile if not provided
        if profile is None:
            profiler = DataProfiler(self.config)
            profile = profiler.profile_dataset(df, target_col)
            self.warnings.extend(profiler.warnings)
        
        # Build pipeline steps based on profile
        steps = []
        
        # Step 1: Remove duplicates
        if profile.get('duplicates', {}).get('n_duplicates', 0) > 0:
            steps.append(('remove_duplicates', DuplicateRemover()))
            self.preprocessing_steps.append("Remove duplicate rows")
        
        # Step 2: Handle missing values
        if profile.get('missing', {}).get('total_missing', 0) > 0:
            steps.append(('handle_missing', MissingValueHandler(self.config)))
            self.preprocessing_steps.append("Impute missing values")
        
        # Step 3: Remove constant features
        if len(profile.get('cardinality', {}).get('low_variance_columns', [])) > 0:
            steps.append(('remove_constant', ConstantFeatureRemover(self.config)))
            self.preprocessing_steps.append("Remove constant/low-variance features")
        
        # Step 4: Handle high cardinality
        if len(profile.get('cardinality', {}).get('high_cardinality_columns', [])) > 0:
            steps.append(('handle_high_cardinality', HighCardinalityHandler(self.config)))
            self.preprocessing_steps.append("Handle high-cardinality features")
        
        # Step 5: Collapse rare categories
        steps.append(('collapse_rare', RareCategoryCollapser(self.config)))
        self.preprocessing_steps.append("Collapse rare categories")
        
        # Step 6: Convert data types
        steps.append(('convert_types', DataTypeConverter()))
        self.preprocessing_steps.append("Convert data types")
        
        # Step 7: Handle outliers (for regression tasks mainly)
        task_type = profile.get('target', {}).get('task_type', 'classification')
        if task_type == 'regression':
            steps.append(('handle_outliers', OutlierHandler(self.config, method='iqr')))
            self.preprocessing_steps.append("Cap outliers (IQR method)")
        
        # Step 8: Encode categorical variables
        steps.append(('encode_categorical', CategoricalEncoder(
            max_categories=self.config.MAX_CATEGORIES_OHE
        )))
        self.preprocessing_steps.append("Encode categorical variables")
        
        # Step 9: Scale numeric features
        steps.append(('scale_numeric', NumericScaler(method='standard')))
        self.preprocessing_steps.append("Scale numeric features")
        
        # Step 10: Feature engineering (if enabled)
        if self.config.ENABLE_AUTO_FEATURE_ENGINEERING:
            n_features = len([col for col in df.columns if col != target_col])
            if n_features <= 20:  # Only for small datasets
                steps.append(('feature_engineering', FeatureEngineer(
                    create_interactions=False,
                    create_polynomials=True
                )))
                self.preprocessing_steps.append("Create engineered features")
        
        # Build pipeline
        self.pipeline = Pipeline(steps)
        
        self.warnings.append(f"INFO: Built preprocessing pipeline with {len(steps)} steps")
        
        return self.pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform data"""
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X, y)
        
        # Convert back to DataFrame if necessary
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        X_transformed = self.pipeline.transform(X)
        
        # Convert back to DataFrame if necessary
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation"""
        if self.pipeline is None:
            return []
        
        # This is simplified - in production you'd track feature names through pipeline
        return []
    
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Complete data preparation workflow
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y, warnings)
        """
        self.warnings = []
        
        # Validate target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Check for missing values in target
        target_missing = df[target_col].isnull().sum()
        if target_missing > 0:
            self.warnings.append(
                f"WARNING: Dropping {target_missing} rows with missing target values"
            )
            df = df[df[target_col].notna()].copy()
        
        # Separate features and target
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        # Remove potential leakage columns (IDs, etc.)
        leakage_keywords = ['id', 'key', 'index', 'uuid', 'guid']
        leakage_cols = []
        for col in X.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in leakage_keywords):
                cardinality_ratio = X[col].nunique() / len(X)
                if cardinality_ratio > 0.95:
                    leakage_cols.append(col)
        
        if leakage_cols:
            self.warnings.append(f"INFO: Removing potential ID columns: {leakage_cols}")
            X = X.drop(columns=leakage_cols)
        
        # Check final dataset size
        if len(X) < self.config.MIN_ROWS_REQUIRED:
            raise ValueError(
                f"Too few rows after cleaning ({len(X)}) - "
                f"need at least {self.config.MIN_ROWS_REQUIRED}"
            )
        
        if len(X.columns) == 0:
            raise ValueError("No features left after preprocessing!")
        
        self.warnings.append(f"✓ Data prepared: {len(X)} rows × {len(X.columns)} features")
        
        return X, y, self.warnings
