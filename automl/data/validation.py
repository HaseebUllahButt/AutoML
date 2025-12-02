"""Column type validator"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re


class ColumnValidator:
    """Validates and fixes columns with mixed types or encoding issues"""
    
    def __init__(self):
        self.warnings = []
    
    def validate_and_fix_column(self, series: pd.Series, col_name: str) -> Tuple[pd.Series, str]:
        """
        Validate and fix a column
        
        Returns:
            Tuple of (fixed_series, inferred_type)
        """
        # Try to detect and fix numeric columns with units or noise
        if series.dtype == 'object':
            fixed, dtype = self._try_fix_numeric_with_noise(series, col_name)
            if dtype == 'numeric':
                return fixed, dtype
            
            # Try to fix boolean
            fixed, dtype = self._try_fix_boolean(series, col_name)
            if dtype == 'boolean':
                return fixed, dtype
        
        return series, str(series.dtype)
    
    def _try_fix_numeric_with_noise(self, series: pd.Series, col_name: str) -> Tuple[pd.Series, str]:
        """Try to clean and convert numeric columns with noise"""
        sample = series.dropna().head(100)
        
        # Check if it looks numeric
        numeric_count = 0
        for val in sample:
            if self._looks_numeric(str(val)):
                numeric_count += 1
        
        if numeric_count / len(sample) < 0.7:
            return series, 'categorical'
        
        # Try to clean
        try:
            cleaned = series.apply(self._clean_numeric_string)
            converted = pd.to_numeric(cleaned, errors='coerce')
            
            # If most values converted successfully
            if converted.notna().sum() / len(series.dropna()) > 0.8:
                self.warnings.append(f"INFO: Converted '{col_name}' to numeric (cleaned units/formatting)")
                return converted, 'numeric'
        except:
            pass
        
        return series, 'categorical'
    
    def _looks_numeric(self, value: str) -> bool:
        """Check if string looks numeric"""
        # Remove common patterns
        cleaned = re.sub(r'[,$%€£¥\s]', '', value)
        cleaned = cleaned.replace(',', '.')  # Handle European decimals
        
        try:
            float(cleaned)
            return True
        except:
            return False
    
    def _clean_numeric_string(self, value):
        """Clean numeric string"""
        if pd.isna(value):
            return np.nan
        
        value = str(value).strip()
        
        # Remove currency symbols and units
        value = re.sub(r'[,$%€£¥]', '', value)
        
        # Remove units (kg, ft, etc.)
        value = re.sub(r'[a-zA-Z]+$', '', value)
        
        # Handle thousands separators and decimal commas
        if ',' in value and '.' in value:
            # Both present - assume , is thousand separator
            value = value.replace(',', '')
        elif ',' in value:
            # Check if it's decimal comma (European) or thousand separator
            if value.count(',') == 1 and len(value.split(',')[1]) <= 2:
                value = value.replace(',', '.')
            else:
                value = value.replace(',', '')
        
        # Remove whitespace
        value = value.replace(' ', '')
        
        try:
            return float(value)
        except:
            return np.nan
    
    def _try_fix_boolean(self, series: pd.Series, col_name: str) -> Tuple[pd.Series, str]:
        """Try to convert to boolean"""
        unique_vals = set(str(v).lower().strip() for v in series.dropna().unique())
        
        true_vals = {'true', '1', 'yes', 't', 'y', 'on'}
        false_vals = {'false', '0', 'no', 'f', 'n', 'off'}
        
        if unique_vals.issubset(true_vals | false_vals):
            converted = series.apply(lambda x: self._to_bool(x, true_vals, false_vals))
            self.warnings.append(f"INFO: Converted '{col_name}' to boolean")
            return converted, 'boolean'
        
        return series, 'categorical'
    
    def _to_bool(self, value, true_vals, false_vals):
        """Convert value to boolean"""
        if pd.isna(value):
            return np.nan
        
        val_str = str(value).lower().strip()
        if val_str in true_vals:
            return 1
        elif val_str in false_vals:
            return 0
        else:
            return np.nan
