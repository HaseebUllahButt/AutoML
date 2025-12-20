"""
CS-245 AutoML System for Classification
Complete Implementation with All Requirements
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
import time
import tempfile
import warnings
from typing import Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc

from automl.config.settings import AutoMLConfig
from automl.data.ingestion import DataIngestor
from automl.data.profiling import DataProfiler
from automl.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from automl.models.trainer import ModelTrainer
from automl.reports.report_generator import ReportGenerator

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="CS-245 AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CLEAN STYLING ====================
st.markdown("""
<style>
    /* MAIN APP - WHITE BACKGROUND, BLACK TEXT */
    .stApp {
        background: #ffffff !important;
    }
    
    body, html {
        background: #ffffff !important;
    }
    
    /* TEXT VISIBILITY - CAREFUL OVERRIDES */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1d2e !important;
        font-weight: 700 !important;
    }
    
    /* Standard text elements */
    p, li, label, .stMarkdown, .stText {
        color: #000000 !important;
    }
    
    /* Ensure markdown container text is black */
    [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f5f5f5 !important;
        padding: 8px !important;
        border-radius: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #000000 !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e0e0e0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Buttons - Preserve White Text */
    .stButton button {
        background: #2196F3 !important;
        color: #ffffff !important;
        padding: 12px 32px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background: #1976D2 !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton button p {
        color: #ffffff !important;
    }
    
    /* Dropdowns & Selects */
    [data-baseweb="select"] div {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"]:hover {
        background: #e3f2fd !important;
        color: #000000 !important;
    }
    
    [aria-selected="true"][role="option"] {
        background: #2196F3 !important;
        color: #ffffff !important;
    }
    
    /* Inputs */
    input, textarea, select {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Code blocks */
    code {
        color: #d63384 !important;
        background: #f8f9fa !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #f5f5f5 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stMetric"] label {
        color: #555555 !important;
    }
    
    [data-testid="stMetric"] div {
        color: #000000 !important;
    }
    
    /* SIDEBAR */
    .stSidebar {
        background: #f8f9fa !important;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] p, 
    .stSidebar label {
        color: #000000 !important;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: #000000 !important;
    }

    [data-testid="stFileUploadDropzone"] {
        background: #f0f2f6 !important;
    }

    
    /* Markdown containers */
    [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background: #f8f9fa !important;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
    }
    
    /* Info/warning/error boxes */
    .stAlert {
        background: #f8f9fa !important;
    }
    
    .stAlert * {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session():
    defaults = {
        'config': AutoMLConfig(),
        'df': None,
        'profile': None,
        'issues': [],
        'preprocessing_config': {},
        'X_processed': None,
        'y': None,
        'pipeline': None,
        'training_results': [],
        'best_model': None,
        'best_model_name': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ==================== MAIN APP ====================
def main():
    st.title("🤖 CS-245 AutoML System")
    st.markdown("### Automated Classification Pipeline")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        st.markdown("### Preprocessing")
        missing_strategy = st.selectbox(
            "Missing Value Imputation",
            ["median", "mean", "mode", "constant"],
            help="Strategy for handling missing values"
        )
        
        scaler_type = st.selectbox(
            "Feature Scaling",
            ["StandardScaler", "MinMaxScaler", "None"],
            help="Method to scale numerical features"
        )
        
        encoding_method = st.selectbox(
            "Categorical Encoding",
            ["One-Hot", "Ordinal"],
            help="Method to encode categorical variables"
        )
        
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data for testing"
        ) / 100
        
        st.session_state.preprocessing_config = {
            'missing_strategy': missing_strategy,
            'scaler': scaler_type,
            'encoding': encoding_method,
            'test_size': test_size
        }
        
        st.markdown("---")
        st.markdown("### Model Training")
        fast_mode = st.checkbox("Fast Mode", help="Train only quick models")
        
        st.markdown("---")
        st.caption("CS-245 ML Project | AutoML System")
    
    # Main tabs
    tabs = st.tabs([
        "📤 Upload",
        "📊 EDA & Issues",
        "⚙️ Preprocessing",
        "🎯 Training",
        "📈 Results",
        "🔮 Prediction",
        "💾 Export"
    ])
    
    # TAB 1: UPLOAD
    with tabs[0]:
        st.markdown("## Dataset Upload")
        st.markdown("Upload a CSV file for classification")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                with st.spinner("Loading dataset..."):
                    ingestor = DataIngestor(st.session_state.config)
                    df, messages = ingestor.ingest(tmp_path)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.success(f"✅ Loaded: {len(df):,} rows × {df.shape[1]} columns")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", df.shape[1])
                        with col3:
                            mem = df.memory_usage(deep=True).sum() / (1024**2)
                            st.metric("Memory", f"{mem:.1f} MB")
                        
                        st.markdown("### Preview")
                        st.dataframe(df.head(20), width="stretch")
                        
                        st.markdown("### Column Types")
                        types_df = pd.DataFrame({
                            'Column': df.dtypes.index,
                            'Type': df.dtypes.values.astype(str)
                        })
                        st.dataframe(types_df, width="stretch")
                    else:
                        st.error("Failed to load dataset")
                
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # TAB 2: EDA & ISSUES
    with tabs[1]:
        st.markdown("## Exploratory Data Analysis & Issue Detection")
        
        if st.session_state.df is None:
            st.info("Please upload a dataset first")
        else:
            df = st.session_state.df
            
            # Target selection
            target_col = st.selectbox("Select Target Column", df.columns.tolist())
            
            if st.button("🔍 Analyze Dataset", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        profiler = DataProfiler(st.session_state.config)
                        profile = profiler.profile_dataset(df, target_col)
                        st.session_state.profile = profile
                        st.session_state.profile['target_column'] = target_col
                    except Exception as e:
                        st.error(f"Analysis Failed: {str(e)}")
                        st.info("Check your dataset for inconsistencies or invalid formats.")
                        st.session_state.profile = None
                        st.stop()
                    
                    # Detect issues
                    issues = []
                    
                    # Feasibility Checks (NEW)
                    n_rows = len(df)
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    n_features = len(numerical_cols) + len(categorical_cols)
                    
                    if n_rows < 50:
                        issues.append({
                            'type': 'Dataset Too Small',
                            'severity': 'High',
                            'description': f"Only {n_rows} samples. ML models need more data to learn patterns effectively.",
                            'fix': "Collect more data (recommended >50 samples minimal)"
                        })
                    
                    if n_features > n_rows:
                        issues.append({
                            'type': 'High Dimensionality',
                            'severity': 'High',
                            'description': f"More features ({n_features}) than samples ({n_rows}). High risk of overfitting.",
                            'fix': "Reduce features or collect more data"
                        })
                    
                    # Missing values check removed - handled in Tab 3 (Preprocessing)
                    
                    # Outliers
                    # numerical_cols is already defined above for feasibility checks
                    outlier_count = 0
                    for col in numerical_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                        outlier_count += outliers
                    
                    if outlier_count > 0:
                        issues.append({
                            'type': 'Outliers',
                            'severity': 'Medium',
                            'count': outlier_count,
                            'description': f"{outlier_count} outliers detected using IQR method",
                            'fix': "Apply capping or removal"
                        })
                    
                    # Class imbalance
                    if profile['target']['task_type'] == 'classification':
                        class_dist = pd.Series(profile['target']['class_distribution'])
                        imbalance_ratio = class_dist.max() / class_dist.min() if class_dist.min() > 0 else float('inf')
                        if imbalance_ratio > 3:
                            issues.append({
                                'type': 'Class Imbalance',
                                'severity': 'Medium',
                                'count': 0,
                                'description': f"Imbalance ratio: {imbalance_ratio:.1f}:1",
                                'fix': "Consider resampling techniques"
                            })
                    
                    st.session_state.issues = issues
                    st.success("✅ Analysis complete!")
            
            if st.session_state.profile:
                profile = st.session_state.profile
                
                # Summary stats
                try:
                    st.markdown("### 📋 Dataset Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", f"{profile['basic']['n_rows']:,}")
                    with col2:
                        st.metric("Features", profile['basic']['n_cols'])
                    with col3:
                        st.metric("Memory (MB)", f"{profile['basic']['memory_mb']:.1f}")
                    with col4:
                        st.metric("Target Classes", profile['target'].get('n_unique', 'N/A'))
                    
                    # Class distribution
                    if 'class_distribution' in profile['target']:
                        st.markdown("### 🎯 Target Distribution")
                        class_dist = pd.Series(profile['target']['class_distribution'])
                        fig, ax = plt.subplots(figsize=(10, 4))
                        class_dist.plot(kind='bar', ax=ax, color='#2196F3')
                        ax.set_title('Class Distribution')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Missing values
                    st.markdown("### 🔍 Missing Values")
                    missing = df.isnull().sum()
                    missing_df = pd.DataFrame({
                        'Feature': missing.index,
                        'Missing': missing.values,
                        'Percent': (missing / len(df) * 100).round(2).values
                    })
                    missing_df = missing_df[missing_df['Missing'] > 0]
                    
                    if len(missing_df) > 0:
                        st.dataframe(missing_df, width="stretch")
                    else:
                        st.success("✅ No missing values")
                    
                    # Outliers
                    st.markdown("### 📐 Outlier Detection (IQR Method)")
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    
                    if len(numerical_cols) == 0:
                        st.info("No numerical columns found for outlier detection/plots.")
                    else:
                        outlier_data = []
                        for col in numerical_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                            if outliers > 0:
                                outlier_data.append({
                                    'Feature': col,
                                    'Outliers': outliers,
                                    'Percent': round(outliers / len(df) * 100, 2)
                                })
                        
                        if outlier_data:
                            st.dataframe(pd.DataFrame(outlier_data), width="stretch")
                        else:
                            st.success("✅ No outliers detected")
                        
                        # Correlation matrix
                        if len(numerical_cols) > 1:
                            st.markdown("### 🔗 Correlation Matrix")
                            corr = df[numerical_cols].corr()
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                                       square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
                            ax.set_title('Feature Correlations')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # Distribution plots
                        st.markdown("### 📊 Numerical Distributions")
                        display_cols = list(numerical_cols)[:4]
                        cols_per_row = 2
                        for i in range(0, len(display_cols), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col_name in enumerate(display_cols[i:i+cols_per_row]):
                                with cols[j]:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    df[col_name].hist(bins=30, ax=ax, color='#2196F3', edgecolor='black')
                                    ax.set_title(col_name, fontweight='bold')
                                    ax.set_xlabel('Value')
                                    ax.set_ylabel('Frequency')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                    
                    # Categorical plots
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        st.markdown("### 📊 Categorical Distributions")
                        for cat_col in list(categorical_cols)[:3]:
                            value_counts = df[cat_col].value_counts().head(10)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            
                            # Fix for matplotlib < 3.2 returning axes
                            value_counts.plot(kind='bar', ax=ax, color='#059669', edgecolor='black')
                            
                            ax.set_title(f'{cat_col} (Top 10)')
                            ax.set_ylabel('Count')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                except Exception as e:
                    st.error(f"Error rendering visualizations: {str(e)}")
                
                # Issues & user approval
                if st.session_state.issues:
                    st.markdown("### ⚠️ Detected Issues & Suggested Fixes")
                    for issue in st.session_state.issues:
                        severity_color = {
                            'High': '🔴',
                            'Medium': '🟡',
                            'Low': '🟢'
                        }
                        with st.expander(f"{severity_color[issue['severity']]} {issue['type']} ({issue['severity']} Priority)"):
                            st.write(f"**Description:** {issue['description']}")
                            st.info(f"**Suggested Fix:** {issue['fix']}")
                            if st.button(f"✅ Approve Fix", key=f"fix_{issue['type']}"):
                                st.success(f"Fix approved for {issue['type']}")
    
    # TAB 3: PREPROCESSING
    with tabs[2]:
        st.markdown("## Preprocessing Pipeline")
        
        if st.session_state.profile is None:
            st.info("Complete EDA first")
        else:
            # Intelligent Missing Value Handling Section
            st.markdown("### 🧹 Data Cleaning & Preprocessing")
            
            # Identify missing columns
            missing_cols = st.session_state.profile.get('missing', {}).get('missing_pct_by_column', {})
            actual_missing_cols = {k: v for k, v in missing_cols.items() if v > 0}
            
            if actual_missing_cols:
                st.warning(f"⚠️ Found {len(actual_missing_cols)} columns with missing values.")
                
                # Global Threshold Setting
                threshold = st.slider(
                    "Drop Column Threshold (%)", 
                    min_value=10, max_value=90, value=50, step=5,
                    help="Columns with missing values above this percentage will be recommended for deletion."
                )
                
                st.markdown("#### Intelligent Handling Suggestions")
                
                # Create a form-like structure for decisions
                handling_decisions = {}
                
                # Display table with controls
                cols = st.columns([2, 1, 2])
                cols[0].markdown("**Column**")
                cols[1].markdown("**Missing %**")
                cols[2].markdown("**Action**")
                
                for col_name, pct in actual_missing_cols.items():
                    c1, c2, c3 = st.columns([2, 1, 2])
                    
                    # Recommendation logic
                    recommendation = "Drop Column" if pct > threshold else "Impute"
                    rec_icon = "🗑️" if pct > threshold else "💊"
                    
                    c1.write(f"{col_name}")
                    c2.write(f"{pct}%")
                    
                    # Action selector
                    decision = c3.selectbox(
                        f"Action for {col_name}",
                        options=["Impute (Median/Mode)", "Drop Column", "Drop Rows"],
                        index=1 if recommendation == "Drop Column" else 0,
                        key=f"missing_action_{col_name}",
                        help=f"AI Recommendation: {rec_icon} {recommendation}"
                    )
                    handling_decisions[col_name] = decision
                
                st.markdown("---")
            else:
                st.success("✅ No missing values detected in dataset.")
                handling_decisions = {}

            st.markdown("### Pipeline Configuration")
            config = st.session_state.preprocessing_config
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing Value Strategy", config['missing_strategy'])
                st.metric("Scaler", config['scaler'])
            with col2:
                st.metric("Encoding", config['encoding'])
                st.metric("Test Split", f"{config['test_size']*100:.0f}%")
            
            if st.button("⚙️ Build Pipeline", type="primary"):
                with st.spinner("Building..."):
                    try:
                        # Apply user decisions first
                        temp_df = st.session_state.df.copy()
                        dropped_cols = []
                        
                        for col, action in handling_decisions.items():
                            if action == "Drop Column":
                                if col in temp_df.columns:
                                    temp_df = temp_df.drop(columns=[col])
                                    dropped_cols.append(col)
                            elif action == "Drop Rows":
                                temp_df = temp_df.dropna(subset=[col])
                        
                        if dropped_cols:
                            st.info(f"Dropped columns based on your selection: {dropped_cols}")
                        
                        # Proceed with build
                        target_col = st.session_state.profile['target_column']
                        builder = PreprocessingPipelineBuilder(st.session_state.config)
                        
                        # Validate data before building
                        if temp_df is None or temp_df.empty:
                            st.error("Dataset is empty after cleaning.")
                        elif target_col not in temp_df.columns:
                             st.error(f"Target column '{target_col}' was dropped! Cannot proceed.")
                        else:
                            X, y, _ = builder.prepare_data(temp_df, target_col)
                            
                            # Encode target if categorical
                            if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
                                encoder = LabelEncoder()
                                y = encoder.fit_transform(y)
                                st.session_state.label_encoder = encoder
                                st.info(f"ℹ️ Target variable encoded: {list(encoder.classes_)}")
                            elif 'label_encoder' in st.session_state:
                                del st.session_state['label_encoder']
                                
                            pipeline = builder.build_pipeline(X, target_col, st.session_state.profile)
                            X_processed = builder.fit_transform(X, y)
                            
                            st.session_state.X_processed = X_processed
                            st.session_state.y = y
                            st.session_state.pipeline = pipeline
                            
                            st.success("✅ Pipeline built!")
                            X, y, _ = builder.prepare_data(st.session_state.df, target_col)
                            
                            # Encode target if categorical
                            if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
                                encoder = LabelEncoder()
                                y = encoder.fit_transform(y)
                                st.session_state.label_encoder = encoder
                                st.info(f"ℹ️ Target variable encoded: {list(encoder.classes_)}")
                            elif 'label_encoder' in st.session_state:
                                del st.session_state['label_encoder']
                                
                            pipeline = builder.build_pipeline(X, target_col, st.session_state.profile)
                            X_processed = builder.fit_transform(X, y)
                            
                            st.session_state.X_processed = X_processed
                            st.session_state.y = y
                            st.session_state.pipeline = pipeline
                            
                            st.success("✅ Pipeline built!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Input Features", len(X.columns))
                            with col2:
                                st.metric("Output Features", X_processed.shape[1])
                            with col3:
                                st.metric("Samples", len(X_processed))
                    except Exception as e:
                        st.error(f"Failed to build pipeline: {str(e)}")
                        st.error("Tip: Check if your dataset contains mixed types or unsupported characters.")
    
    # TAB 4: TRAINING
    with tabs[3]:
        st.markdown("## Model Training")
        
        if st.session_state.X_processed is None:
            st.info("Build preprocessing pipeline first")
        else:
            if st.button("🎯 Train Models", type="primary"):
                trainer = ModelTrainer(st.session_state.config)
                task_type = st.session_state.profile['target']['task_type']
                
                progress = st.progress(0)
                status = st.empty()
                
                def update_progress(msg, pct):
                    if pct is not None:
                        progress.progress(int(pct * 100))
                    # Use markdown with black color ensuring visibility
                    status.markdown(f"<p style='color:black;'>{msg}</p>", unsafe_allow_html=True)
                
                try:
                    results = trainer.train_models(
                        st.session_state.X_processed,
                        st.session_state.y,
                        task_type=task_type,
                        fast_only=fast_mode,
                        progress_callback=update_progress
                    )
                    
                    st.session_state.training_results = results['all_results']
                    st.session_state.best_model = trainer.best_model
                    st.session_state.best_model_name = trainer.best_model_name
                    
                    progress.progress(100)
                    status.success("✅ Training complete!")
                    
                    if trainer.best_model_name:
                        st.markdown(f"### 🏆 Best Model: {trainer.best_model_name}")
                        st.metric("Score", f"{trainer.best_score:.4f}")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    # TAB 5: RESULTS
    with tabs[4]:
        st.markdown("## Model Comparison & Results")
        
        if st.session_state.training_results:
            results = st.session_state.training_results
            successful = [r for r in results if r.get('status') == 'success']
            
            if successful:
                # Metrics table
                st.markdown("### 📊 Model Comparison")
                
                task_type = st.session_state.profile['target'].get('task_type', 'classification')
                
                # DEBUG: Remove after fixing
                # st.info(f"DEBUG: Detected Task Type = {task_type}")
                
                display_data = []
                
                for r in successful:
                    metrics = r.get('metrics', {})
                    if task_type == 'classification':
                        display_data.append({
                            'Model': r.get('model_name'),
                            'Accuracy': metrics.get('test_accuracy', 0),
                            'F1-Score': metrics.get('test_f1', 0),
                            'Precision': metrics.get('test_precision', 0),
                            'Recall': metrics.get('test_recall', 0),
                            'Time (s)': r.get('training_time', 0)
                        })
                    else:
                        display_data.append({
                            'Model': r.get('model_name'),
                            'R2 Score': metrics.get('test_r2', 0),
                            'MAE': metrics.get('test_mae', 0),
                            # 'MSE': metrics.get('test_mse', 0), # Not calculated in trainer
                            'RMSE': metrics.get('test_rmse', 0),
                            'Time (s)': r.get('training_time', 0)
                        })
                
                results_df = pd.DataFrame(display_data)
                
                if task_type == 'classification':
                    results_df = results_df.sort_values('Accuracy', ascending=False)
                else:
                    results_df = results_df.sort_values('R2 Score', ascending=False)
                    
                st.dataframe(results_df, width="stretch")
                
                # Download CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    "model_results.csv",
                    "text/csv",
                    type="primary"
                )
                
                # Metric comparison bar chart
                st.markdown("### 📈 Metric Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(results_df))
                
                if task_type == 'classification':
                    width = 0.2
                    ax.bar(x - width, results_df['Accuracy'], width, label='Accuracy', color='#2196F3')
                    ax.bar(x, results_df['F1-Score'], width, label='F1-Score', color='#4CAF50')
                    ax.bar(x + width, results_df['Precision'], width, label='Precision', color='#FF9800')
                    ax.set_ylabel('Score (0-1)')
                else:
                    width = 0.35
                    # For regression, we primarily plot R2 as it's 0-1 scale similar to classification metrics
                    # MAE/MSE scales vary too much to plot on same axis easily without dual axis
                    ax.bar(x, results_df['R2 Score'], width, label='R2 Score', color='#2196F3')
                    ax.set_ylabel('R2 Score')
                    
                ax.set_xlabel('Models')
                ax.set_title(f'Model Performance ({task_type.title()})')
                ax.set_xticks(x)
                ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Confusion matrix for best model (Classification only)
                # We check the MODEL itself, not just the profile, to be sure.
                from sklearn.base import is_classifier
                is_clf_model = is_classifier(st.session_state.best_model) or \
                               hasattr(st.session_state.best_model, 'classes_') or \
                               hasattr(st.session_state.best_model, 'predict_proba')
                
                if st.session_state.best_model and is_clf_model:
                    st.markdown("### 🎯 Best Model Confusion Matrix")
                    try:
                        trainer = ModelTrainer(st.session_state.config)
                        trainer.best_model = st.session_state.best_model
                        
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            st.session_state.X_processed,
                            st.session_state.y,
                            test_size=0.2,
                            random_state=42
                        )
                        
                        y_pred = st.session_state.best_model.predict(X_test)
                        
                        # Ensure y_test and y_pred are compatible types for confusion_matrix
                        # If regression model crept in, y_pred might be float.
                        if y_pred.dtype.kind in 'fc': # float or complex
                            y_pred = y_pred.round().astype(int)
                            
                        # Double check y_test too
                        y_test_clean = y_test
                        if hasattr(y_test, 'dtype') and y_test.dtype.kind in 'fc':
                             y_test_clean = y_test.round().astype(int)
                        
                        cm = confusion_matrix(y_test_clean, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title('Confusion Matrix')
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                        
                        # Set tick labels if encoder exists
                        if 'label_encoder' in st.session_state:
                            try:
                                classes = st.session_state.label_encoder.classes_
                                if len(classes) == cm.shape[0]:
                                    ax.set_xticklabels(classes, rotation=45, ha='right')
                                    ax.set_yticklabels(classes, rotation=0)
                            except:
                                pass
                                
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Could not generate confusion matrix: {e}")
                    
                # ROC Curve (Binary Classification Only)
                if len(np.unique(st.session_state.y)) == 2 and task_type == 'classification':
                    st.markdown("### 📈 ROC Curves")
                    st.info("ROC (Receiver Operating Characteristic) curves show the trade-off between true positive rate and false positive rate")
                    
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import roc_curve, auc
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state.X_processed,
                        st.session_state.y,
                        test_size=0.2,
                        random_state=42
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    for r in successful:
                        model = r.get('model')
                        if model and hasattr(model, 'predict_proba'):
                            try:
                                y_proba = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_proba)
                                roc_auc = auc(fpr, tpr)
                                ax.plot(fpr, tpr, label=f"{r['model_name']} (AUC={roc_auc:.3f})", linewidth=2)
                            except:
                                pass
                    
                    # Random classifier baseline
                    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=2)
                    ax.set_xlabel('False Positive Rate', fontsize=12)
                    ax.set_ylabel('True Positive Rate', fontsize=12)
                    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
                    ax.legend(loc='lower right')
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Feature Importance (Tree-based Models)
                if st.session_state.best_model:
                    best_model = st.session_state.best_model
                    if hasattr(best_model, 'feature_importances_'):
                        st.markdown("### 🎯 Feature Importance")
                        st.info("Shows which features contribute most to predictions (higher = more important)")
                        
                        # Get feature names
                        try:
                            # Get original feature names before encoding
                            X_original = st.session_state.df.drop(columns=[st.session_state.profile['target_column']])
                            n_features = best_model.feature_importances_.shape[0]
                            
                            # Create feature names (approximation)
                            if n_features <= len(X_original.columns):
                                feature_names = X_original.columns[:n_features]
                            else:
                                feature_names = [f'Feature_{i}' for i in range(n_features)]
                            
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': best_model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(15)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(importance_df['Feature'], importance_df['Importance'], color='#2196F3')
                            ax.set_xlabel('Importance', fontsize=12)
                            ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
                            ax.invert_yaxis()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.warning(f"Could not display feature importance: {str(e)}")
        else:
            st.info("Train models first")
    
    # TAB 6: EXPORT
    with tabs[5]:
        st.markdown("## Export Results")
        
        if st.session_state.best_model:
            # Executive Summary
            st.markdown("## 📋 Executive Summary")
            
            with st.expander("View Summary Before Generating Report", expanded=True):
                profile = st.session_state.profile
                successful_results = [r for r in st.session_state.training_results if r.get('status') == 'success']
                
                # Dataset info
                st.markdown("### 📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", f"{profile['basic']['n_rows']:,}")
                with col2:
                    st.metric("Features", profile['basic']['n_cols'])
                with col3:
                    st.metric("Task Type", profile['target']['task_type'].title())
                
                # Best model info
                st.markdown("### 🏆 Best Performing Model")
                best_result = max(successful_results, key=lambda x: x['metrics'].get('test_accuracy', 0))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model", best_result['model_name'])
                with col2:
                    metric_name = 'test_accuracy' if profile['target']['task_type'] == 'classification' else 'test_r2'
                    st.metric("Performance", f"{best_result['metrics'].get(metric_name, 0):.4f}")
                with col3:
                    st.metric("Training Time", f"{best_result['training_time']:.2f}s")
                
                # Key insights
                st.markdown("### 💡 Key Insights")
                insights = []
                
                # Data quality - use DataFrame directly instead of profile
                df = st.session_state.df
                missing_total = df.isnull().sum().sum()
                
                if missing_total > 0:
                    insights.append(f"⚠️ Dataset has {missing_total} missing values across features")
                else:
                    insights.append("✅ No missing values detected")
                
                # Model performance
                if best_result['metrics'].get('test_accuracy', 0) > 0.9:
                    insights.append("✅ Excellent model performance (>90% accuracy)")
                elif best_result['metrics'].get('test_accuracy', 0) > 0.8:
                    insights.append("✅ Good model performance (>80% accuracy)")
                else:
                    insights.append("⚠️ Model performance could be improved")
                
                # Model count
                insights.append(f"✅ Successfully trained {len(successful_results)} out of {len(st.session_state.training_results)} models")
                
                for insight in insights:
                    st.write(insight)
                
                # Recommendations
                st.markdown("### 🎯 Recommendations")
                recommendations = []
                
                if best_result['model_name'] in ['random_forest', 'decision_tree']:
                    recommendations.append("Consider feature importance analysis to identify key predictors")
                
                if len(np.unique(st.session_state.y)) == 2:
                    recommendations.append("Review ROC curves for detailed performance analysis")
                
                if missing_total > 0:
                    recommendations.append("Investigate missing value patterns for potential data collection improvements")
                
                recommendations.append("Deploy best model for production use")
                recommendations.append("Monitor model performance on new data regularly")
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
    
    # TAB 5: PREDICTION PLAYGROUND
    with tabs[5]:
        st.markdown("## 🔮 Prediction Playground")
        st.markdown("Test your model with 'What-If' scenarios")
        
        if st.session_state.best_model is None:
            st.info("Please train a model first (Tab 4)")
        else:
            model_name = st.session_state.get('best_model_name', 'Unknown Model')
            st.success(f"Using best model: **{model_name}**")
            
            # Dynamic Input Widgets
            st.markdown("### 1. Enter Input Values")
            
            if st.session_state.df is not None:
                # Get feature columns (excluding target)
                target_col = st.session_state.profile['target_column']
                feature_cols = [c for c in st.session_state.df.columns if c != target_col]
                
                # Create form for inputs
                with st.form("prediction_form"):
                    input_data = {}
                    
                    # Create 2 columns for widgets
                    cols_sw = st.columns(2)
                    
                    for i, col in enumerate(feature_cols):
                        with cols_sw[i % 2]:
                            # Determine column type from original dataframe
                            dtype = st.session_state.df[col].dtype
                            unique_vals = st.session_state.df[col].unique()
                            
                            if pd.api.types.is_numeric_dtype(dtype):
                                # Numerical Input
                                min_val = float(st.session_state.df[col].min())
                                max_val = float(st.session_state.df[col].max())
                                mean_val = float(st.session_state.df[col].mean())
                                
                                input_data[col] = st.number_input(
                                    f"{col}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    help=f"Range: {min_val} - {max_val}"
                                )
                            else:
                                # Categorical Input
                                input_data[col] = st.selectbox(
                                    f"{col}",
                                    options=unique_vals,
                                    help=f"{len(unique_vals)} options"
                                )
                    
                    submitted = st.form_submit_button("🚀 Predict", type="primary")
                    
                    if submitted:
                        st.markdown("### 2. Result")
                        try:
                            # Create DataFrame from input
                            input_df = pd.DataFrame([input_data])
                            
                            # Preprocess
                            # We need to apply the same transformations
                            # Note: The pipeline in session_state handles this transform
                            if hasattr(st.session_state.pipeline, 'transform'):
                                # For some simpler pipelines or if pipeline was just the ColumnTransformer
                                try:
                                    X_pred = st.session_state.pipeline.transform(input_df)
                                except Exception as e_trans:
                                    # Fallback: if pipeline expects X and y or other issues, try to use the raw input 
                                    # if the model pipeline includes preprocessing (which it usually does in sklearn)
                                    # BUT here our 'pipeline' is likely just the preprocessor.
                                    # Let's try standard transform.
                                    # If 'fit_transform' was used on X_processed, then 'pipeline' should be the fitted preprocessor.
                                    st.warning(f"Preprocessing warning: {str(e_trans)}")
                                    X_pred = input_df
                            else:
                                X_pred = input_df
                                
                            # Predict
                            prediction = st.session_state.best_model.predict(X_pred)[0]
                            
                            # Decode prediction if encoded
                            prediction_label = prediction
                            if 'label_encoder' in st.session_state:
                                try:
                                    prediction_label = st.session_state.label_encoder.inverse_transform([prediction])[0]
                                except:
                                    pass
                            
                            # Display result
                            st.success(f"**Prediction:** {prediction_label}")
                            
                            # Probability (if classification)
                            if hasattr(st.session_state.best_model, 'predict_proba'):
                                proba = st.session_state.best_model.predict_proba(X_pred)[0]
                                classes = st.session_state.best_model.classes_
                                
                                # Decode classes if encoded
                                classes_labels = classes
                                if 'label_encoder' in st.session_state:
                                    try:
                                        classes_labels = st.session_state.label_encoder.inverse_transform(classes)
                                    except:
                                        pass
                                
                                st.markdown("#### Confidence Scores")
                                for c, p in zip(classes_labels, proba):
                                    st.write(f"**{c}**: {p:.1%}")
                                    st.progress(float(p))
                                    
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")
                            st.info("Tip: Ensure inputs match the format of training data.")

    # TAB 6: RESULTS (Export)
    with tabs[6]:
        # Report generation
        st.markdown("## 💾 Export & Reports")
        
        st.markdown("---")
        if st.button("📄 Generate Full Report", type="primary"):
            if st.session_state.best_model:
                with st.spinner("⏳ Generating comprehensive report..."):
                    try:
                        report_gen = ReportGenerator(st.session_state.config)
                        report_path = report_gen.generate_report(
                            profile=st.session_state.profile,
                            preprocessing_steps=[str(step) for step in st.session_state.pipeline.steps] if st.session_state.pipeline else [],
                            training_results=st.session_state.training_results,
                            target_col=st.session_state.profile.get('target_column', 'target')
                        )
                        
                        st.success(f"✅ Report generated: {report_path}")
                        
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                "⬇️ Download PDF Report",
                                f,
                                file_name=os.path.basename(report_path),
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            else:
                st.info("Train models first to generate report")

if __name__ == "__main__":
    main()
