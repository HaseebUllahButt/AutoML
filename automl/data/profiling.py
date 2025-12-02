"""
Comprehensive data profiling and quality analysis
Detects all data quality issues, leakage, outliers, and generates detailed reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from collections import Counter
import re

from ..config.settings import AutoMLConfig


class DataProfiler:
    """Analyzes dataset quality and generates comprehensive warnings"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.profile = {}
        self.warnings = []
        self.recommendations = []
        
    def profile_dataset(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive data profile
        
        Args:
            df: Input DataFrame
            target_col: Name of target column (if known)
            
        Returns:
            Dictionary containing all profiling information
        """
        self.warnings = []
        self.recommendations = []
        
        self.profile = {
            'basic': self._profile_basic(df),
            'missing': self._profile_missing(df),
            'dtypes': self._profile_dtypes(df),
            'cardinality': self._profile_cardinality(df),
            'duplicates': self._profile_duplicates(df),
            'outliers': self._profile_outliers(df),
            'statistics': self._profile_statistics(df),
            'correlations': self._profile_correlations(df),
        }
        
        if target_col and target_col in df.columns:
            self.profile['target'] = self._profile_target(df, target_col)
            self.profile['leakage'] = self._profile_leakage(df, target_col)
        
        self.profile['warnings'] = self.warnings
        self.profile['recommendations'] = self.recommendations
        
        return self.profile
    
    def _profile_basic(self, df: pd.DataFrame) -> Dict:
        """Basic dataset statistics"""
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        if memory_mb > self.config.MEMORY_WARNING_THRESHOLD_MB:
            self.warnings.append(f"WARNING: Large dataset ({memory_mb:.1f}MB) - may cause memory issues")
        
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': round(memory_mb, 2),
            'columns': list(df.columns),
            'shape': df.shape,
        }
    
    def _profile_missing(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values with detailed warnings"""
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(2)
        
        # Identify problematic columns
        high_missing_cols = missing_pct[missing_pct > 50].to_dict()
        very_high_missing_cols = missing_pct[missing_pct > self.config.MAX_MISSING_RATIO * 100].to_dict()
        
        if very_high_missing_cols:
            self.warnings.append(
                f"CRITICAL: {len(very_high_missing_cols)} columns have >{self.config.MAX_MISSING_RATIO*100}% missing "
                f"(will be dropped): {list(very_high_missing_cols.keys())}"
            )
        elif high_missing_cols:
            self.warnings.append(
                f"WARNING: {len(high_missing_cols)} columns have >50% missing values: "
                f"{list(high_missing_cols.keys())}"
            )
        
        # Check rows with many missing values
        missing_per_row = df.isnull().sum(axis=1)
        rows_mostly_missing = (missing_per_row / len(df.columns) > 0.5).sum()
        
        if rows_mostly_missing > 0:
            self.warnings.append(
                f"WARNING: {rows_mostly_missing} rows have >50% missing values (may be dropped)"
            )
        
        return {
            'total_missing': int(missing_count.sum()),
            'missing_by_column': missing_count.to_dict(),
            'missing_pct_by_column': missing_pct.to_dict(),
            'high_missing_columns': high_missing_cols,
            'very_high_missing_columns': very_high_missing_cols,
            'rows_with_high_missing': int(rows_mostly_missing),
        }
    
    def _profile_dtypes(self, df: pd.DataFrame) -> Dict:
        """Analyze data types and detect mixed types"""
        dtype_info = {}
        mixed_type_cols = []
        numeric_stored_as_object = []
        
        for col in df.columns:
            pandas_dtype = str(df[col].dtype)
            inferred_type = self._infer_column_type(df[col])
            
            dtype_info[col] = {
                'pandas_dtype': pandas_dtype,
                'inferred_type': inferred_type,
                'n_unique': df[col].nunique(),
            }
            
            # Detect mixed types
            if inferred_type == 'mixed':
                mixed_type_cols.append(col)
                self.warnings.append(f"WARNING: Column '{col}' has mixed data types")
            
            # Detect numbers stored as strings
            if pandas_dtype == 'object' and inferred_type == 'numeric_with_noise':
                numeric_stored_as_object.append(col)
                self.warnings.append(
                    f"WARNING: Column '{col}' appears numeric but has non-numeric values "
                    f"(e.g., '10k', '5+', 'N/A')"
                )
        
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
        
        return {
            'dtype_info': dtype_info,
            'mixed_type_columns': mixed_type_cols,
            'numeric_stored_as_object': numeric_stored_as_object,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'n_numeric': len(numeric_cols),
            'n_categorical': len(categorical_cols),
            'n_datetime': len(datetime_cols),
        }
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the true type of a column"""
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return 'empty'
        
        # If already numeric, return numeric
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Try to convert to numeric
        try:
            pd.to_numeric(non_null, errors='raise')
            return 'numeric'
        except (ValueError, TypeError):
            # Check if mostly numeric with some noise
            numeric_count = pd.to_numeric(non_null, errors='coerce').notna().sum()
            if numeric_count / len(non_null) > 0.8:
                return 'numeric_with_noise'
        
        # Try to convert to datetime
        try:
            pd.to_datetime(non_null, errors='raise', infer_datetime_format=True)
            return 'datetime'
        except:
            pass
        
        # Check if mixed types
        type_counts = non_null.apply(type).value_counts()
        if len(type_counts) > 1:
            return 'mixed'
        
        # Check if boolean
        unique_vals = set(str(v).lower().strip() for v in non_null.unique())
        if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}):
            return 'boolean'
        
        return 'categorical'
    
    def _profile_cardinality(self, df: pd.DataFrame) -> Dict:
        """Analyze cardinality and detect potential ID columns"""
        cardinality = {}
        high_cardinality_cols = []
        potential_id_cols = []
        low_variance_cols = []
        
        for col in df.columns:
            n_unique = df[col].nunique()
            n_total = len(df)
            cardinality_ratio = n_unique / n_total if n_total > 0 else 0
            
            cardinality[col] = {
                'n_unique': n_unique,
                'cardinality_ratio': round(cardinality_ratio, 4),
            }
            
            # Detect potential ID columns (very high cardinality)
            if cardinality_ratio > self.config.MAX_CARDINALITY_RATIO and n_unique > 100:
                potential_id_cols.append(col)
                self.warnings.append(
                    f"WARNING: Column '{col}' has very high cardinality "
                    f"({n_unique} unique values, {cardinality_ratio:.1%}) - likely an ID column"
                )
            
            # Detect high cardinality categoricals
            if df[col].dtype == 'object' and n_unique > self.config.MAX_CATEGORIES_OHE:
                high_cardinality_cols.append(col)
                self.recommendations.append(
                    f"INFO: Column '{col}' has {n_unique} categories - "
                    f"will use target encoding or hashing instead of one-hot"
                )
            
            # Detect low variance (constant or near-constant)
            if n_unique == 1:
                low_variance_cols.append(col)
                self.warnings.append(f"WARNING: Column '{col}' is constant (only 1 unique value)")
            elif cardinality_ratio < 0.01 and n_unique < 5:
                low_variance_cols.append(col)
                self.recommendations.append(
                    f"INFO: Column '{col}' has very low variance ({n_unique} unique values)"
                )
        
        return {
            'cardinality': cardinality,
            'high_cardinality_columns': high_cardinality_cols,
            'potential_id_columns': potential_id_cols,
            'low_variance_columns': low_variance_cols,
        }
    
    def _profile_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate rows"""
        n_duplicates = df.duplicated().sum()
        duplicate_ratio = n_duplicates / len(df) if len(df) > 0 else 0
        
        if n_duplicates > 0:
            self.warnings.append(
                f"WARNING: {n_duplicates} duplicate rows found ({duplicate_ratio:.1%})"
            )
            self.recommendations.append("INFO: Duplicate rows will be removed during preprocessing")
        
        return {
            'n_duplicates': int(n_duplicates),
            'duplicate_ratio': round(duplicate_ratio, 4),
        }
    
    def _profile_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 3:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > self.config.OUTLIER_Z_THRESHOLD).sum()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((data < Q1 - self.config.OUTLIER_IQR_MULTIPLIER * IQR) | 
                           (data > Q3 + self.config.OUTLIER_IQR_MULTIPLIER * IQR)).sum()
            
            outlier_info[col] = {
                'z_score_outliers': int(z_outliers),
                'iqr_outliers': int(iqr_outliers),
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
            }
            
            # Check for unrealistic values
            if col.lower() in ['age'] and (data.min() < 0 or data.max() > 150):
                self.warnings.append(f"WARNING: Column '{col}' has unrealistic values (min={data.min()}, max={data.max()})")
            
            if iqr_outliers > len(data) * 0.1:
                self.warnings.append(f"WARNING: Column '{col}' has many outliers ({iqr_outliers}, {iqr_outliers/len(data):.1%})")
        
        return outlier_info
    
    def _profile_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate statistical summaries"""
        numeric_summary = df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        
        categorical_summary = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            value_counts = df[col].value_counts().head(10).to_dict()
            categorical_summary[col] = {
                'n_unique': df[col].nunique(),
                'top_values': value_counts,
                'most_common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            }
        
        return {
            'numeric_summary': numeric_summary,
            'categorical_summary': categorical_summary,
        }
    
    def _profile_correlations(self, df: pd.DataFrame) -> Dict:
        """Compute correlations (if dataset isn't too large)"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > self.config.MAX_CORRELATION_FEATURES:
            self.recommendations.append(
                f"INFO: Too many features ({len(numeric_df.columns)}) for correlation matrix - skipping"
            )
            return {'correlation_matrix': None, 'high_correlations': []}
        
        if len(numeric_df.columns) < 2:
            return {'correlation_matrix': None, 'high_correlations': []}
        
        try:
            corr_matrix = numeric_df.corr()
            
            # Find highly correlated pairs
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': round(corr_val, 3),
                        })
            
            if high_corr:
                self.warnings.append(
                    f"WARNING: Found {len(high_corr)} highly correlated feature pairs (|r| > 0.9)"
                )
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_corr,
            }
        except Exception as e:
            self.warnings.append(f"WARNING: Could not compute correlations: {e}")
            return {'correlation_matrix': None, 'high_correlations': []}
    
    def _profile_target(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze target column"""
        target = df[target_col]
        
        # Check for missing values in target
        n_missing = target.isnull().sum()
        if n_missing > 0:
            self.warnings.append(
                f"CRITICAL: Target column '{target_col}' has {n_missing} missing values - "
                f"these rows will be dropped"
            )
        
        # Determine task type
        n_unique = target.nunique()
        
        if pd.api.types.is_numeric_dtype(target) and n_unique > 20:
            task_type = 'regression'
        else:
            task_type = 'classification'
        
        profile = {
            'task_type': task_type,
            'n_unique': n_unique,
            'n_missing': int(n_missing),
        }
        
        if task_type == 'classification':
            value_counts = target.value_counts().to_dict()
            profile['class_distribution'] = value_counts
            profile['is_balanced'] = self._check_balance(value_counts)
            
            # Check for imbalance
            if not profile['is_balanced']:
                self.warnings.append(
                    f"WARNING: Imbalanced classes detected - consider using class weights or resampling"
                )
            
            # Check minimum samples per class
            min_samples = min(value_counts.values())
            if min_samples < self.config.MIN_SAMPLES_PER_CLASS:
                self.warnings.append(
                    f"CRITICAL: Some classes have very few samples (min={min_samples}) - "
                    f"stratification may fail"
                )
        else:
            profile['min'] = float(target.min())
            profile['max'] = float(target.max())
            profile['mean'] = float(target.mean())
            profile['std'] = float(target.std())
            profile['skewness'] = float(target.skew())
        
        return profile
    
    def _check_balance(self, class_distribution: Dict) -> bool:
        """Check if classes are balanced"""
        if len(class_distribution) < 2:
            return True
        
        counts = list(class_distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        
        # Consider balanced if ratio is less than 3:1
        return (max_count / min_count) < 3
    
    def _profile_leakage(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Detect potential data leakage"""
        leakage_suspects = []
        
        target = df[target_col]
        
        for col in df.columns:
            if col == target_col:
                continue
            
            # Check for perfect correlation with target
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target):
                try:
                    corr = df[[col, target_col]].corr().iloc[0, 1]
                    if abs(corr) > 0.99:
                        leakage_suspects.append({
                            'column': col,
                            'reason': f'Perfect correlation with target ({corr:.3f})',
                            'severity': 'high',
                        })
                        self.warnings.append(
                            f"CRITICAL: Column '{col}' has suspiciously high correlation "
                            f"with target ({corr:.3f}) - possible data leakage!"
                        )
                except:
                    pass
            
            # Check for ID-like patterns
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['id', 'key', 'index', 'uuid', 'guid']):
                cardinality_ratio = df[col].nunique() / len(df)
                if cardinality_ratio > 0.95:
                    leakage_suspects.append({
                        'column': col,
                        'reason': 'High cardinality ID column',
                        'severity': 'medium',
                    })
                    self.recommendations.append(
                        f"INFO: Column '{col}' appears to be an ID - will be excluded from training"
                    )
        
        return {
            'leakage_suspects': leakage_suspects,
            'n_suspects': len(leakage_suspects),
        }
