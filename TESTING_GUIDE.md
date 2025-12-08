# 🧪 Error Handling Validation & Testing Guide

## Quick Start Testing

### Test 1: Invalid File Handling
```bash
# Test with non-existent file
python -c "
from automl.data.ingestion import DataIngestor
ingestor = DataIngestor()
df, msgs = ingestor.ingest('/nonexistent/file.csv')
print('Messages:', msgs)
"
# Expected: ERROR messages about file not found
```

### Test 2: Empty File Handling
```bash
# Create empty file and test
touch empty.csv
python -c "
from automl.data.ingestion import DataIngestor
ingestor = DataIngestor()
df, msgs = ingestor.ingest('empty.csv')
print('DF:', df)
print('Messages:', msgs)
"
# Expected: ERROR about empty file
```

### Test 3: Encoding Issue Handling
```python
# Create file with mixed encoding
with open('mixed_encoding.csv', 'wb') as f:
    f.write(b'Name,Age\n')
    f.write('Caf\xc3\xa9,25\n'.encode('utf-8'))
    f.write(b'\xff\xfeW\x00e\x00b\x00,\x002\x006\x00\n')  # UTF-16

from automl.data.ingestion import DataIngestor
ingestor = DataIngestor()
df, msgs = ingestor.ingest('mixed_encoding.csv')
print('Success:', df is not None)
print('Messages:', [m for m in msgs if 'encoding' in m.lower()])
# Expected: Successfully read despite encoding issues
```

### Test 4: Type Validation
```python
from automl.utils import validate_inputs
import pandas as pd

@validate_inputs(X=pd.DataFrame, y=pd.Series)
def test_func(X, y):
    return len(X) + len(y)

# This should work
df = pd.DataFrame({'a': [1, 2, 3]})
s = pd.Series([1, 2, 3])
result = test_func(df, s)
print('Result:', result)

# This should raise DataValidationError
try:
    result = test_func([1, 2, 3], s)  # List instead of DataFrame
except Exception as e:
    print('Error caught:', type(e).__name__)
```

### Test 5: Range Validation
```python
from automl.utils import validate_ranges

@validate_ranges(threshold=(0, 1), max_iter=(1, 1000))
def train_model(threshold=0.5, max_iter=100):
    return f"threshold={threshold}, max_iter={max_iter}"

# Valid values
print(train_model(0.5, 100))

# Invalid values
try:
    train_model(1.5, 100)  # Out of range
except Exception as e:
    print('Error caught:', type(e).__name__)
```

### Test 6: Retry Mechanism
```python
from automl.utils import retry
import random

attempt = 0

@retry(max_attempts=3, delay=0.1, backoff=1.5)
def flaky_operation():
    global attempt
    attempt += 1
    print(f"Attempt {attempt}")
    if attempt < 3:
        raise ValueError("Simulated failure")
    return "Success!"

result = flaky_operation()
print(f"Result after {attempt} attempts: {result}")
```

### Test 7: Error Context
```python
from automl.utils import ErrorContext

# Successful operation
print("Test 1: Success case")
try:
    with ErrorContext("loading data"):
        data = [1, 2, 3, 4, 5]
        print(f"Loaded {len(data)} items")
except Exception as e:
    print(f"Error: {e}")

# Failed operation
print("\nTest 2: Error case")
try:
    with ErrorContext("processing data"):
        data = None
        result = data[0]  # This will fail
except Exception as e:
    print(f"Caught: {type(e).__name__}")
```

### Test 8: ErrorCollector
```python
from automl.utils import ErrorCollector

collector = ErrorCollector(max_errors=5)

# Simulate processing multiple items
items = list(range(10))
for item in items:
    try:
        if item % 3 == 0:
            raise ValueError(f"Item {item} is divisible by 3")
        collector.add_info(f"Processed item {item}")
    except Exception as e:
        collector.add_error(f"Failed on item {item}: {e}")

print(collector.get_summary())
print(f"\nDetails: {collector.to_dict()}")
```

---

## Integration Testing

### Test 9: Full Data Pipeline with Error Handling
```python
from automl.data.ingestion import DataIngestor
from automl.data.profiling import DataProfiler
from automl.config.settings import AutoMLConfig

try:
    config = AutoMLConfig()
    
    # Test ingestion
    print("Step 1: Ingesting data...")
    ingestor = DataIngestor(config)
    df, msgs = ingestor.ingest('sample_data/loan_approval.csv')
    
    if df is None:
        print("Failed to ingest")
        print(f"Errors: {msgs}")
    else:
        print(f"✓ Loaded {len(df)} rows")
        
        # Test profiling with validation
        print("\nStep 2: Profiling data...")
        profiler = DataProfiler(config)
        profile = profiler.profile_dataset(df, target_col='Loan_Status')
        
        print(f"✓ Profile complete")
        print(f"  Warnings: {len(profile.get('warnings', []))}")
        print(f"  Recommendations: {len(profile.get('recommendations', []))}")

except Exception as e:
    print(f"Pipeline failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

### Test 10: Training Pipeline with Error Recovery
```python
from automl.models.trainer import ModelTrainer
from automl.config.settings import AutoMLConfig
import pandas as pd
import numpy as np

try:
    config = AutoMLConfig()
    
    # Create dummy data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    
    # Test with invalid inputs first
    print("Test 1: Invalid input handling")
    trainer = ModelTrainer(config)
    
    try:
        results = trainer.train_models(None, y)
    except Exception as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
    
    # Test with valid inputs
    print("\nTest 2: Valid training")
    results = trainer.train_models(X, y, task_type='classification', fast_only=True)
    
    print(f"✓ Training complete")
    print(f"  Models trained: {len(results.get('all_results', []))}")
    print(f"  Warnings: {len(results.get('warnings', []))}")
    print(f"  Best model: {results.get('best_model_name')}")

except Exception as e:
    print(f"Training failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

---

## Error Scenario Testing

### Scenario 1: Handling Missing Data
```python
import pandas as pd
import numpy as np
from automl.data.profiling import DataProfiler

# Create data with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'C': [1, 2, 3, 4, 5],
    'target': [0, 1, 0, 1, 0]
})

profiler = DataProfiler()
profile = profiler.profile_dataset(df, 'target')

print("Missing value analysis:")
print(profile['missing'])
print("\nWarnings:")
for w in profile['warnings']:
    if 'missing' in w.lower():
        print(f"  {w}")
```

### Scenario 2: Handling Type Conflicts
```python
import pandas as pd
from automl.data.validation import ColumnValidator

# Create data with mixed types
series = pd.Series(['$100', '50€', '25', 'N/A', '1000'])

validator = ColumnValidator()
fixed, dtype = validator.validate_and_fix_column(series, 'Price')

print(f"Original type: {series.dtype}")
print(f"Fixed type: {dtype}")
print(f"Fixed values:\n{fixed}")
print(f"Warnings: {validator.warnings}")
```

### Scenario 3: Large File Handling
```python
import pandas as pd
import tempfile
from automl.data.ingestion import DataIngestor

# Create a moderately large file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    # Write header
    f.write('A,B,C,D,E\n')
    # Write 100k rows
    for i in range(100000):
        f.write(f'{i},{i*2},{i*3},{i*4},{i*5}\n')
    temp_path = f.name

try:
    ingestor = DataIngestor()
    df, msgs = ingestor.ingest(temp_path)
    
    if df is not None:
        print(f"✓ Successfully loaded large file: {len(df)} rows")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    print(f"\nInfo messages:")
    for msg in msgs:
        if 'WARNING' in msg or 'INFO' in msg:
            print(f"  {msg}")

finally:
    import os
    os.unlink(temp_path)
```

---

## Automated Test Suite

Create `test_error_handling.py`:

```python
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from automl.utils.error_handlers import (
    IngestException, ProfilingException, DataValidationError,
    ErrorContext, ErrorCollector, validate_inputs, retry
)
from automl.data.ingestion import DataIngestor
from automl.data.profiling import DataProfiler

class TestErrorHandling(unittest.TestCase):
    
    def test_file_not_found(self):
        """Test handling of non-existent file"""
        ingestor = DataIngestor()
        df, msgs = ingestor.ingest('/nonexistent/file.csv')
        self.assertIsNone(df)
        self.assertTrue(any('ERROR' in msg for msg in msgs))
    
    def test_empty_file(self):
        """Test handling of empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            ingestor = DataIngestor()
            df, msgs = ingestor.ingest(temp_path)
            self.assertIsNone(df)
            self.assertTrue(any('empty' in msg.lower() for msg in msgs))
        finally:
            os.unlink(temp_path)
    
    def test_invalid_dataframe_profiling(self):
        """Test profiling with invalid input"""
        profiler = DataProfiler()
        
        with self.assertRaises(ProfilingException):
            profiler.profile_dataset(None)
    
    def test_validate_inputs_decorator(self):
        """Test input validation decorator"""
        
        @validate_inputs(x=int, y=int)
        def add(x, y):
            return x + y
        
        # Valid call
        result = add(1, 2)
        self.assertEqual(result, 3)
        
        # Invalid call
        with self.assertRaises(DataValidationError):
            add("1", 2)
    
    def test_retry_decorator(self):
        """Test retry mechanism"""
        attempts = {'count': 0}
        
        @retry(max_attempts=3, delay=0.01)
        def flaky_function():
            attempts['count'] += 1
            if attempts['count'] < 3:
                raise ValueError("Simulated failure")
            return "success"
        
        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(attempts['count'], 3)
    
    def test_error_collector(self):
        """Test error collection"""
        collector = ErrorCollector()
        
        collector.add_error("Error 1")
        collector.add_warning("Warning 1")
        collector.add_info("Info 1")
        
        self.assertTrue(collector.has_errors())
        self.assertEqual(collector.error_count(), 1)
        self.assertEqual(collector.warning_count(), 1)

if __name__ == '__main__':
    unittest.main()
```

Run with:
```bash
python -m pytest test_error_handling.py -v
```

---

## Checklist: Error Handling Coverage

- [x] File existence validation
- [x] Empty file detection
- [x] Large file warnings
- [x] Encoding detection & fallback
- [x] Delimiter detection & fallback
- [x] CSV parsing with multiple engines
- [x] Type validation on inputs
- [x] Range validation on parameters
- [x] DataFrame validation in profiling
- [x] Column existence validation
- [x] Target column validation
- [x] Model training error recovery
- [x] Individual model failure handling
- [x] Report generation validation
- [x] File I/O error handling
- [x] Temporary file cleanup
- [x] Session state consistency
- [x] User-friendly error messages
- [x] Comprehensive logging
- [x] Context-aware error information

---

**Status**: ✅ Complete  
**Version**: 1.0  
**Last Updated**: December 7, 2025
