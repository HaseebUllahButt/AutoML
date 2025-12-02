"""
Robust CSV ingestion with extreme edge case handling
Handles all file format nightmares, encoding issues, delimiter problems, etc.
"""

import pandas as pd
import numpy as np
import chardet
import io
import zipfile
import gzip
import warnings
from pathlib import Path
from typing import Tuple, Optional, List
import re

from ..config.settings import AutoMLConfig


class DataIngestor:
    """Handles all file reading nightmares and edge cases"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.warnings = []
        self.info_messages = []
        
    def ingest(self, file_path: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Main ingestion pipeline with comprehensive error handling
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (DataFrame or None, list of warnings/errors/info messages)
        """
        self.warnings = []
        self.info_messages = []
        
        try:
            # 1. Basic file validation
            if not self._validate_file(file_path):
                return None, self.warnings + self.info_messages
            
            # 2. Handle compressed files
            file_path = self._handle_compression(file_path)
            
            # 3. Detect if file is actually HTML/Excel disguised as CSV
            if not self._validate_file_content(file_path):
                return None, self.warnings + self.info_messages
            
            # 4. Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # 5. Detect delimiter
            delimiter = self._detect_delimiter(file_path, encoding)
            
            # 6. Read with all safeguards
            df = self._safe_read(file_path, encoding, delimiter)
            
            if df is None or df.empty:
                self.warnings.append("ERROR: DataFrame is empty after reading")
                return None, self.warnings + self.info_messages
            
            # 7. Post-read validation and cleanup
            df = self._post_read_cleanup(df)
            
            if df is None or len(df) < self.config.MIN_ROWS_REQUIRED:
                self.warnings.append(f"ERROR: Too few rows ({len(df) if df is not None else 0}) - need at least {self.config.MIN_ROWS_REQUIRED}")
                return None, self.warnings + self.info_messages
            
            self.info_messages.append(f"✓ Successfully ingested {len(df)} rows × {len(df.columns)} columns")
            
            return df, self.warnings + self.info_messages
            
        except Exception as e:
            self.warnings.append(f"CRITICAL: Failed to ingest file: {str(e)}")
            return None, self.warnings + self.info_messages
    
    def _validate_file(self, file_path: str) -> bool:
        """Check file exists and isn't too large"""
        path = Path(file_path)
        
        if not path.exists():
            self.warnings.append("ERROR: File does not exist")
            return False
        
        if not path.is_file():
            self.warnings.append("ERROR: Path is not a file")
            return False
        
        size_mb = path.stat().st_size / (1024 * 1024)
        
        if size_mb == 0:
            self.warnings.append("ERROR: File is empty (0 bytes)")
            return False
        
        if size_mb > self.config.MAX_FILE_SIZE_MB:
            self.warnings.append(f"ERROR: File too large ({size_mb:.1f}MB > {self.config.MAX_FILE_SIZE_MB}MB)")
            return False
        
        if size_mb > 100:
            self.warnings.append(f"WARNING: Large file ({size_mb:.1f}MB), processing may be slow")
        
        return True
    
    def _validate_file_content(self, file_path: str) -> bool:
        """Detect if file is actually HTML or Excel disguised as CSV"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1000)
            
            # Check for HTML
            if b'<html' in header.lower() or b'<!doctype' in header.lower():
                self.warnings.append("ERROR: File appears to be HTML, not CSV")
                return False
            
            # Check for Excel (magic numbers)
            if header[:8] == b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':  # Old Excel
                self.warnings.append("ERROR: File is Excel (.xls), not CSV")
                return False
            
            if header[:4] == b'PK\x03\x04':  # New Excel (xlsx) or zip
                # Could be xlsx or legitimate zip
                self.warnings.append("INFO: File appears to be compressed or Excel format")
            
            return True
            
        except Exception as e:
            self.warnings.append(f"WARNING: Could not validate file content: {e}")
            return True  # Continue anyway
    
    def _handle_compression(self, file_path: str) -> str:
        """Detect and decompress if needed"""
        path = Path(file_path)
        
        # Check if it's a zip
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                names = z.namelist()
                if len(names) == 1:
                    self.info_messages.append("INFO: Detected ZIP compression, extracting...")
                    extracted = z.extract(names[0], path.parent)
                    return extracted
                elif len(names) > 1:
                    # Try to find a CSV file
                    csv_files = [n for n in names if n.lower().endswith(('.csv', '.txt', '.tsv'))]
                    if csv_files:
                        self.info_messages.append(f"INFO: ZIP contains multiple files, using {csv_files[0]}")
                        extracted = z.extract(csv_files[0], path.parent)
                        return extracted
                    else:
                        self.warnings.append("ERROR: ZIP contains multiple files, none appear to be CSV")
                        return file_path
        except zipfile.BadZipFile:
            pass
        
        # Check if it's gzip
        try:
            with gzip.open(file_path, 'rb') as f:
                f.read(1)
            self.info_messages.append("INFO: Detected GZIP compression")
            return file_path  # pandas can handle gzip natively
        except:
            pass
        
        return file_path
    
    def _detect_encoding(self, file_path: str) -> str:
        """Try multiple encodings with chardet as fallback"""
        # Quick detection with chardet
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(min(100000, Path(file_path).stat().st_size))  # Sample first 100KB
            
            result = chardet.detect(raw)
            detected = result['encoding']
            confidence = result['confidence']
            
            if confidence > 0.7 and detected:
                self.info_messages.append(f"INFO: Detected encoding '{detected}' (confidence: {confidence:.2f})")
                return detected
        except Exception as e:
            self.warnings.append(f"WARNING: Chardet encoding detection failed: {e}")
        
        # Fallback: try each encoding manually
        for enc in self.config.ENCODING_ATTEMPTS:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.read(10000)  # Try reading 10K chars
                self.info_messages.append(f"INFO: Using encoding '{enc}'")
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        self.warnings.append("WARNING: Could not detect encoding reliably, using utf-8 with error replacement")
        return 'utf-8'
    
    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect delimiter by sampling and counting"""
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Read first 10 lines for sampling
                sample_lines = [f.readline() for _ in range(min(10, sum(1 for _ in open(file_path, 'r', encoding=encoding, errors='replace'))))]
                sample = ''.join(sample_lines)
            
            # Count each delimiter in each line
            delimiter_scores = {}
            for delim in self.config.DELIMITER_ATTEMPTS:
                counts = [line.count(delim) for line in sample_lines if line.strip()]
                if counts:
                    # Good delimiter should have consistent count across lines
                    avg_count = np.mean(counts)
                    std_count = np.std(counts)
                    # Score: high average, low std deviation
                    if avg_count > 0:
                        consistency = avg_count / (std_count + 1)  # Add 1 to avoid division by zero
                        delimiter_scores[delim] = avg_count * consistency
            
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                if delimiter_scores[best_delimiter] > 0:
                    delim_name = {
                        ',': 'comma',
                        ';': 'semicolon',
                        '|': 'pipe',
                        '\t': 'tab',
                        '^': 'caret',
                        '~': 'tilde',
                        ':': 'colon',
                    }.get(best_delimiter, repr(best_delimiter))
                    self.info_messages.append(f"INFO: Detected delimiter: {delim_name}")
                    return best_delimiter
            
            self.warnings.append("WARNING: No delimiter detected reliably, assuming comma")
            return ','
            
        except Exception as e:
            self.warnings.append(f"WARNING: Delimiter detection failed: {e}, assuming comma")
            return ','
    
    def _safe_read(self, file_path: str, encoding: str, delimiter: str) -> Optional[pd.DataFrame]:
        """Read CSV with maximum safety and error handling"""
        read_params = {
            'filepath_or_buffer': file_path,
            'encoding': encoding,
            'delimiter': delimiter,
            'on_bad_lines': 'skip',  # Skip malformed rows
            'encoding_errors': 'replace',  # Replace bad chars with �
            'low_memory': False,
            'na_values': self.config.MISSING_INDICATORS,
            'keep_default_na': True,
        }
        
        # Try reading with C engine first
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(**read_params)
            self.info_messages.append(f"INFO: Read {len(df)} rows with C engine")
            return df
        
        except pd.errors.ParserError as e:
            self.warnings.append(f"WARNING: C engine parser error: {str(e)[:100]}")
            
            # Try with Python engine (more forgiving)
            try:
                read_params['engine'] = 'python'
                read_params['on_bad_lines'] = 'skip'
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_csv(**read_params)
                self.info_messages.append(f"INFO: Read {len(df)} rows with Python engine (fallback)")
                return df
            except Exception as e2:
                self.warnings.append(f"ERROR: Python engine also failed: {str(e2)[:100]}")
                return None
        
        except Exception as e:
            self.warnings.append(f"ERROR: Failed to read CSV: {str(e)[:100]}")
            return None
    
    def _post_read_cleanup(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Comprehensive cleanup after reading"""
        if df is None or df.empty:
            return None
        
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # 1. Remove BOM (Byte Order Mark) from first column name
        if len(df.columns) > 0 and isinstance(df.columns[0], str):
            first_col = df.columns[0]
            cleaned = first_col.replace('\ufeff', '').replace('\ufffe', '')
            if cleaned != first_col:
                df.columns = [cleaned] + list(df.columns[1:])
                self.info_messages.append("INFO: Removed BOM from first column")
        
        # 2. Remove zero-width and other invisible characters from column names
        df.columns = [self._clean_column_name(str(col)) for col in df.columns]
        
        # 3. Handle duplicate column names
        df = self._fix_duplicate_columns(df)
        
        # 4. Remove completely empty rows
        df = df.dropna(how='all')
        if len(df) < original_rows:
            removed = original_rows - len(df)
            self.info_messages.append(f"INFO: Removed {removed} completely empty rows")
        
        # 5. Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        if len(df.columns) < original_cols:
            removed = original_cols - len(df.columns)
            self.info_messages.append(f"INFO: Removed {removed} completely empty columns")
        
        # 6. Remove unnamed columns (often artifacts from Excel)
        unnamed_cols = [col for col in df.columns if str(col).lower().startswith('unnamed:')]
        if unnamed_cols:
            # Check if they're actually empty or just index columns
            cols_to_drop = []
            for col in unnamed_cols:
                if df[col].isna().sum() / len(df) > 0.95:  # >95% missing
                    cols_to_drop.append(col)
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.info_messages.append(f"INFO: Removed {len(cols_to_drop)} unnamed/empty columns")
        
        # 7. Reset index
        df = df.reset_index(drop=True)
        
        # 8. Check for suspiciously uniform structure (might indicate header repeated)
        df = self._remove_repeated_headers(df)
        
        return df
    
    def _clean_column_name(self, name: str) -> str:
        """Remove invisible characters and clean column names"""
        # Remove zero-width and invisible Unicode characters
        invisible_chars = [
            '\u200b',  # Zero width space
            '\u200c',  # Zero width non-joiner
            '\u200d',  # Zero width joiner
            '\ufeff',  # BOM
            '\ufffe',  # Reverse BOM
            '\u00a0',  # Non-breaking space
            '\u2028',  # Line separator
            '\u2029',  # Paragraph separator
        ]
        
        for char in invisible_chars:
            name = name.replace(char, '')
        
        # Strip leading/trailing whitespace
        name = name.strip()
        
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        
        # If name is now empty, generate a placeholder
        if not name:
            name = 'column'
        
        return name
    
    def _fix_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate column names by appending suffixes"""
        cols = df.columns.tolist()
        seen = {}
        new_cols = []
        
        for col in cols:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
                new_cols.append(new_col)
                self.warnings.append(f"WARNING: Duplicate column '{col}' renamed to '{new_col}'")
            else:
                seen[col] = 0
                new_cols.append(col)
        
        df.columns = new_cols
        return df
    
    def _remove_repeated_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove rows that are actually repeated headers"""
        if len(df) < 2:
            return df
        
        # Check if any row matches the header
        header_str = [str(col).lower().strip() for col in df.columns]
        rows_to_drop = []
        
        for idx, row in df.iterrows():
            row_str = [str(val).lower().strip() for val in row.values]
            if row_str == header_str:
                rows_to_drop.append(idx)
        
        if rows_to_drop:
            df = df.drop(index=rows_to_drop)
            self.info_messages.append(f"INFO: Removed {len(rows_to_drop)} repeated header rows")
        
        return df
