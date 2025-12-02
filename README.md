# 🤖 AutoML System

**A fully automated machine learning pipeline that handles real-world data nightmares**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

This AutoML system is a **production-ready**, **classical ML** pipeline (no deep learning, no LLMs) that handles **50+ edge cases** automatically.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

This will open the AutoML system in your browser at `http://localhost:8501`

### 3. Use the System

1. **Upload Data** (Tab 1): Drop your CSV file
2. **Profile Data** (Tab 2): Select target column and analyze quality
3. **Preprocess** (Tab 3): Auto-build preprocessing pipeline
4. **Train Models** (Tab 4): Train multiple models automatically
5. **Export** (Tab 5): Download model and HTML report

---

## 🧪 Quick Test

Create a test dataset and try it:

```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'salary': np.random.randint(30000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], n),
    'loan_approved': np.random.choice([0, 1], n, p=[0.3, 0.7])
})

# Add some edge cases
df.loc[0, 'age'] = '25 years'  # Number with unit
df.loc[np.random.choice(df.index, 50), 'credit_score'] = np.nan  # Missing values

df.to_csv('sample_data/test_data.csv', index=False)
print("✓ Test dataset created!")
```

Then run:
```bash
streamlit run app.py
```

Upload `sample_data/test_data.csv` and watch the magic happen! ✨

---

## 💻 Programmatic Usage

```python
from automl.data.ingestion import DataIngestor
from automl.data.profiling import DataProfiler
from automl.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from automl.models.trainer import ModelTrainer

# 1. Load data
ingestor = DataIngestor()
df, messages = ingestor.ingest('your_data.csv')

# 2. Profile
profiler = DataProfiler()
profile = profiler.profile_dataset(df, target_col='target')

# 3. Preprocess
builder = PreprocessingPipelineBuilder()
X, y, warnings = builder.prepare_data(df, 'target')
pipeline = builder.build_pipeline(X, 'target', profile)
X_processed = builder.fit_transform(X, y)

# 4. Train
trainer = ModelTrainer()
results = trainer.train_models(X_processed, y)
print(f"Best model: {trainer.best_model_name} (Score: {trainer.best_score})")

# 5. Predict
predictions = trainer.predict(X_new_processed)

# 6. Save
trainer.save_model('model.pkl')
```

---

## 📁 Project Structure

```
AutoML/
├── app.py                          # Streamlit UI (START HERE!)
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── automl/                         # Core package
│   ├── config/settings.py         # Configuration
│   ├── data/                      # Data ingestion & profiling
│   ├── preprocessing/             # Cleaning & encoding
│   ├── models/                    # Model training
│   ├── reports/                   # Report generation
│   └── utils/                     # Utilities
│
├── outputs/                       # Generated files
└── sample_data/                   # Test datasets
```

---

## 🎯 Key Capabilities

### Data Ingestion
- ✅ Multiple encodings (UTF-8, Latin-1, UTF-16)
- ✅ Various delimiters (`,`, `;`, `|`, `\t`)
- ✅ Compressed files (ZIP, GZIP)
- ✅ Malformed CSVs
- ✅ Mixed data types
- ✅ Missing value detection (50+ formats)

### Preprocessing
- ✅ Smart missing value imputation
- ✅ Outlier handling
- ✅ Categorical encoding
- ✅ Feature scaling
- ✅ Duplicate removal
- ✅ Data leakage detection

### Models
- **Classification**: Logistic, Random Forest, XGBoost, LightGBM, etc.
- **Regression**: Linear, Ridge, XGBoost, LightGBM, etc.
- Automatic hyperparameter tuning
- Cross-validation
- Model comparison

### Reports
- Beautiful HTML reports
- Data quality analysis
- Model performance metrics
- Recommendations

---

## 🛠️ Troubleshooting

### Installation Issues
```bash
# If you get import errors, reinstall:
pip install --upgrade -r requirements.txt
```

### Streamlit Not Opening
```bash
# Check if Streamlit is installed:
streamlit --version

# If not, install it:
pip install streamlit

# Run again:
streamlit run app.py
```

### Large Files
Edit `automl/config/settings.py`:
```python
MAX_FILE_SIZE_MB = 1000  # Increase this
```

---

## 📊 Performance

- **Small** (<1K rows): 10-30 seconds
- **Medium** (1K-100K rows): 1-5 minutes  
- **Large** (100K-1M rows): 5-30 minutes

---

## 📚 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- streamlit, chardet

See `requirements.txt` for complete list.

---

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

---

## 📄 License

MIT License

---

**Built with ❤️ using pure classical ML**

No deep learning. No LLMs. Just solid engineering. 🚀
