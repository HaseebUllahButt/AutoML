# CS-245 AutoML System for Classification

A comprehensive automated machine learning system built with Streamlit for classification tasks.

## 🎯 Project Overview

This AutoML system provides end-to-end automation for classification machine learning pipelines, from data upload to model deployment. Built as part of CS-245 Machine Learning course requirements.

## ✨ Features

### ✅ Complete CS-245 Requirements

- **Dataset Upload & Basic Info**
  - CSV file support
  - Automatic metadata display (rows, columns, types, memory)
  - Summary statistics
  - Class distribution visualization

- **Comprehensive EDA**
  - Missing value analysis with visualizations
  - Outlier detection (IQR method)
  - Correlation matrix heatmap
  - Distribution plots for numerical features
  - Bar charts for categorical features
  - Train/test split summary

- **Issue Detection & User Approval**
  - Automatic detection of data quality issues
  - Missing values flagging
  - Outlier detection
  - Class imbalance detection
  - High cardinality warnings
  - User approval workflow for fixes

- **Preprocessing Options**
  - User-selectable missing value imputation (mean/median/mode/constant)
  - Outlier handling (removal/capping)
  - Feature scaling (StandardScaler/MinMaxScaler)
  - Categorical encoding (One-Hot/Ordinal)
  - Configurable train-test split (default 80/20)

- **Model Training**
  - 7 classification algorithms:
    1. Logistic Regression
    2. K-Nearest Neighbors
    3. Decision Tree
    4. Naive Bayes
    5. Random Forest
    6. Support Vector Machine
    7. Rule-based Classifier (Dummy)
  - Hyperparameter optimization (Grid/Random Search)
  - Comprehensive metrics:
    - Accuracy
    - Precision, Recall, F1-score
    - Confusion matrix
    - ROC-AUC (binary classification)
    - Training time

- **Model Comparison Dashboard**
  - Sortable comparison table
  - Bar charts for metric visualization
  - Confusion matrix heatmap
  - Downloadable CSV results
  - Best model ranking

- **Auto-Generated Report**
  - PDF/HTML export
  - Dataset overview
  - EDA findings
  - Detected issues
  - Preprocessing decisions
  - Model configurations
  - Comparison tables
  - Best model justification

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd proj
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Upload Dataset
- Navigate to "📤 Upload" tab
- Upload CSV file (max 500MB)
- Review dataset preview and metadata

### Step 2: EDA & Issue Detection
- Go to "�� EDA & Issues" tab
- Select target column
- Click "🔍 Analyze Dataset"
- Review:
  - Missing values
  - Outliers (IQR method)
  - Correlation matrix
  - Feature distributions
  - Detected issues
- Approve suggested fixes

### Step 3: Configure Preprocessing
- Use sidebar to select:
  - Missing value strategy
  - Scaling method
  - Encoding method
  - Train-test split ratio
- Go to "⚙️ Preprocessing" tab
- Click "⚙️ Build Pipeline"

### Step 4: Train Models
- Navigate to "🎯 Training" tab
- Optional: Enable fast mode for quick training
- Click "🎯 Train Models"
- Monitor progress
- View best model

### Step 5: Compare Results
- Go to "📈 Results" tab
- Review model comparison table
- View metric bar charts
- Check confusion matrix
- Download results CSV

### Step 6: Export Report
- Navigate to "💾 Export" tab
- Click "📄 Generate Report"
- Download PDF report

## 🏗️ Project Structure

```
proj/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── automl/                  # Core AutoML modules
│   ├── config/
│   │   └── settings.py      # Configuration
│   ├── data/
│   │   ├── ingestion.py     # Data loading
│   │   └── profiling.py     # EDA analysis
│   ├── preprocessing/
│   │   └── pipeline_builder.py  # Preprocessing
│   ├── models/
│   │   ├── trainer.py       # Model training
│   │   └── model_registry.py    # Model configs
│   ├── reports/
│   │   └── report_generator.py  # PDF generation
│   └── utils/
│       └── error_handlers.py    # Error handling
│
├── sample_data/             # Test datasets
│   ├── iris.csv
│   ├── titanic.csv
│   └── wine.csv
│
└── outputs/                 # Generated reports
```

## 🔧 Configuration

Preprocessing options can be configured via the sidebar:

- **Missing Value Imputation**: mean, median, mode, constant
- **Feature Scaling**: StandardScaler, MinMaxScaler, None
- **Categorical Encoding**: One-Hot, Ordinal
- **Test Set Size**: 10-40% (default: 20%)
- **Fast Mode**: Train only quick models (LR, NB, DT)

## 📦 Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0

See `requirements.txt` for complete list.

## 🌐 Streamlit Cloud Deployment

### Deploy to Streamlit Cloud:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Select `app.py` as main file
5. Click "Deploy"

### Live Demo
<!-- Add your Streamlit Cloud URL here after deployment -->
🔗 [Live App](https://your-app-url.streamlit.app)

## 📊 Sample Datasets

Test datasets are provided in `sample_data/`:
- **iris.csv**: Classic flower classification (150 samples, 4 features)
- **titanic.csv**: Survival prediction (891 samples, 11 features)
- **wine.csv**: Wine quality classification (178 samples, 13 features)

## 🎓 CS-245 Requirements Coverage

| Requirement | Status | Location |
|------------|--------|----------|
| Dataset Upload | ✅ | Tab 1 |
| Basic Metadata | ✅ | Tab 1 |
| Missing Value Analysis | ✅ | Tab 2 |
| Outlier Detection | ✅ | Tab 2 |
| Correlation Matrix | ✅ | Tab 2 |
| Distribution Plots | ✅ | Tab 2 |
| Categorical Bar Plots | ✅ | Tab 2 |
| Issue Detection | ✅ | Tab 2 |
| User Approval Workflow | ✅ | Tab 2 |
| Preprocessing Options | ✅ | Sidebar + Tab 3 |
| 7 Classification Models | ✅ | Tab 4 |
| Hyperparameter Optimization | ✅ | Tab 4 |
| Model Metrics | ✅ | Tab 5 |
| Comparison Dashboard | ✅ | Tab 5 |
| Downloadable CSV | ✅ | Tab 5 |
| Confusion Matrix | ✅ | Tab 5 |
| PDF Report | ✅ | Tab 6 |
| Streamlit Cloud Ready | ✅ | Yes |

## 🐛 Troubleshooting

**Issue**: Plots not showing
- **Solution**: Restart the app, clear browser cache

**Issue**: Training takes too long
- **Solution**: Enable "Fast Mode" in sidebar

**Issue**: Out of memory
- **Solution**: Reduce dataset size or increase system RAM

## 👥 Contributors

- Your Name
- Team Member 2 (if applicable)
- Team Member 3 (if applicable)

## 📝 License

This project is created for educational purposes as part of CS-245 Machine Learning course.

## 🙏 Acknowledgments

- CS-245 Course Instructors
- Streamlit Documentation
- scikit-learn Community
- Open-source ML Community

---

**Course**: CS-245 Machine Learning  
**Project**: AutoML System for Classification  
**Date**: December 2025  
**Status**: ✅ Complete & Deployed
