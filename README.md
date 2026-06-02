# 🤖 Production-Grade AutoML System for Classification & Regression

A comprehensive, clean-architecture Automated Machine Learning (AutoML) platform designed to automate end-to-end machine learning pipelines. Built with a modular Python backend and an interactive Streamlit frontend, this system handles messy, real-world datasets with extreme edge-case resilience and zero-config deployment.
---

## 🏗️ Clean Architecture Overview

This project is built using decoupled software design patterns. It separates data validation, profiling, preprocessing, model registry, hyperparameter tuning, and reporting into distinct modules adhering to SOLID principles.

```mermaid
graph TD
    A[Raw CSV/TSV File] --> B[Data Ingestor]
    B -->|BOM, Encodings, Delimiters Handled| C[Data Profiler]
    C -->|EDA & Quality Profiling| D[Issue Detector]
    D -->|User Approval Workflow| E[Preprocessing Pipeline Builder]
    E -->|Clean, Encoded & Scaled Data| F[Model Trainer]
    F -->|Model Registry / HP Tuning| G[Evaluation Dashboard]
    G -->|Comparison Metrics, Confusion Matrix| H[Best Model & Standalone HTML Report]
```

---

## 🌟 Core Features & Engineering Highlights

### 1. 📥 Robust Data Ingestion
- **Automatic Encoding Detection**: Leverages `chardet` alongside multi-encoding attempts (`utf-8`, `latin-1`, etc.) to read files without Unicode errors.
- **Smart Delimiter Detection**: Samples file lines and uses statistical consistency (mean/std ratio) to automatically detect commas, semicolons, tabs, and pipes.
- **Compression & Format Support**: Seamlessly extracts and reads `.zip` and `.gzip` files, and automatically detects if an Excel or HTML file is incorrectly disguised with a `.csv` extension.
- **Pre-Clean Safeguards**: Detects and strips Byte Order Marks (BOM), removes duplicate/empty column names, and discards repeated headers.

### 2. 🔍 Automated Profiling & Data Quality Checks
- **Exploratory Data Analysis (EDA)**: Automatic calculation of dataset statistics, missing value ratios, and outlier ratios.
- **Visualizations**: Matplotlib & Seaborn-powered correlation matrix heatmaps, numerical distribution histograms, and categorical frequency bar charts.
- **Intelligent Issue Detection**: Identifies critical dataset concerns:
  - **Dataset Feasibility**: High-dimensionality warnings ($features > samples$) and small sample counts.
  - **Class Imbalance**: Flags skew in class distribution.
  - **Outliers & Cardinality**: Detects extreme values and flags high-cardinality columns.
- **User-in-the-Loop Fixes**: A dedicated Streamlit interface allowing users to review and approve suggested preprocessing fixes.

### 3. ⚙️ Preprocessing Pipeline
- **Dynamic Imputation**: Custom strategies for mean, median, mode, or constant imputation.
- **Currency & Unit Cleaning**: Automatically detects currency symbols and units of measurement (e.g., "kg", "ft", "$") in string columns, parses them, and cleans them into floating point values.
- **Categorical Encoders**: Smart One-Hot and Ordinal encoding with automatic rare category collapsing.
- **Feature Leakage Prevention**: Identifies and automatically drops identifier columns (e.g., `id`, `uuid`, `index`).
- **Feature Scaling**: Configurable Standard and Min-Max scaling.

### 4. 🎯 Model Registry & Trainer Suite
- **Comprehensive Algorithm Support**: Fits up to 9 classification and 11 regression algorithms:
  - Linear/Logistic Regression, Random Forests, Gradient Boosting (AdaBoost, XGBoost, LightGBM), K-Nearest Neighbors, SVMs, and Naive Bayes.
  - Baseline `Dummy` classifiers and regressors are used as control variables to measure exact value add.
- **Randomized Hyperparameter Tuning**: Automatically optimizes models via randomized grid search cross-validation.
- **Resource Guardrails**: Enforces time limits (`MAX_TRAINING_TIME_SECONDS`), limits training rows on large datasets, and manages parallel job counts to prevent memory overflows.

### 5. 📊 Evaluation Dashboard & Reporting
- **Metric Dashboards**: Shows sortable comparison tables of training time, accuracy, precision, recall, F1-score, RMSE, MAE, and $R^2$.
- **Interactive Visuals**: Confusion matrices, class distribution plots, and correlation matrices.
- **Automated Report Generation**: Generates standalone HTML executive summaries detailing the data quality profile, preprocessing steps, training configurations, and next steps for loading the model.

---

## 📁 Project Directory Structure

```
AutoML/
├── app.py                      # Interactive Streamlit Web Interface
├── requirements.txt            # Python Dependencies
├── README.md                   # System Documentation
│
├── automl/                     # Core Backend Framework
│   ├── config/
│   │   └── settings.py        # Centralized system settings and thresholds
│   ├── data/
│   │   ├── ingestion.py       # Robust file ingestion and format resolution
│   │   ├── profiling.py       # Statistical profiling and EDA
│   │   └── validation.py      # Column-level validation & unit/currency cleaner
│   ├── preprocessing/
│   │   ├── cleaners.py        # Missing value, outlier, and variance cleaners
│   │   ├── encoders.py        # Scalers, categorical encoders, and feature engineering
│   │   └── pipeline_builder.py# Scikit-learn Pipeline orchestrator
│   ├── models/
│   │   ├── trainer.py         # Cross-validation, tuning, and evaluation
│   │   └── model_registry.py  # Model configurations and parameter spaces
│   ├── reports/
│   │   └── report_generator.py# HTML report generator engine
│   └── utils/
│       ├── error_handlers.py  # Customized exceptions and error boundaries
│       └── logger.py          # Unified logger settings
│
└── sample_data/               # Pre-bundled datasets (Iris, Titanic, Wine)
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository:
```bash
git clone https://github.com/HaseebUllahButt/AutoML.git
cd AutoML
```

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Interface
Launch the Streamlit app:
```bash
streamlit run app.py
```
The interface will automatically load at `http://localhost:8501`.

---

## 👥 Contributors

- **Haseeb Ullah Butt**
- **Ali Mubashir**
---

## 📝 License

This project is created for educational purposes as part of CS-245 Machine Learning course.
