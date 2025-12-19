"""
CS-245 AutoML System for Classification
Complete Implementation with All Requirements
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import os
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
    
    /* Text colors */
    h1, h2, h3 {
        color: #1a1d2e !important;
        font-weight: 700 !important;
    }
    
    p, div, span, li {
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
    
    /* Buttons */
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
    
    /* Dropdowns */
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
    
    /* Inputs and selects */
    input, textarea, select {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* All labels */
    label, [data-testid="stWidgetLabel"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #f8f9fa !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stMetric"] * {
        color: #000000 !important;
    }
    
    /* SIDEBAR */
    .stSidebar {
        background: #f8f9fa !important;
    }
    
    .stSidebar * {
        color: #000000 !important;
    }
    
    .stSidebar h2, .stSidebar h3 {
        color: #1a1d2e !important;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: #f8f9fa !important;
        padding: 20px !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] * {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: #ffffff !important;
        border: 2px dashed #2196F3 !important;
        border-radius: 8px !important;
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
        'training_results': None,
        'best_model': None,
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
                    profiler = DataProfiler(st.session_state.config)
                    profile = profiler.profile_dataset(df, target_col)
                    st.session_state.profile = profile
                    st.session_state.profile['target_column'] = target_col
                    
                    # Detect issues
                    issues = []
                    
                    # Missing values
                    missing = df.isnull().sum()
                    if missing.sum() > 0:
                        issues.append({
                            'type': 'Missing Values',
                            'severity': 'High',
                            'count': int(missing.sum()),
                            'description': f"{missing.sum()} total missing values across {(missing > 0).sum()} features",
                            'fix': f"Impute using {missing_strategy}"
                        })
                    
                    # Outliers
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
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
                        value_counts.plot(kind='bar', ax=ax, color='#059669', edgecolor='black')
                        ax.set_title(f'{cat_col} (Top 10)')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                
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
            st.markdown("### Selected Configuration")
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
                    target_col = st.session_state.profile['target_column']
                    builder = PreprocessingPipelineBuilder(st.session_state.config)
                    
                    X, y, _ = builder.prepare_data(st.session_state.df, target_col)
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
                    status.text(msg)
                
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
                display_data = []
                for r in successful:
                    metrics = r.get('metrics', {})
                    display_data.append({
                        'Model': r.get('model_name'),
                        'Accuracy': metrics.get('test_accuracy', 0),
                        'F1-Score': metrics.get('test_f1', 0),
                        'Precision': metrics.get('test_precision', 0),
                        'Recall': metrics.get('test_recall', 0),
                        'Time (s)': r.get('training_time', 0)
                    })
                
                results_df = pd.DataFrame(display_data)
                results_df = results_df.sort_values('Accuracy', ascending=False)
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
                width = 0.2
                ax.bar(x - width, results_df['Accuracy'], width, label='Accuracy', color='#2196F3')
                ax.bar(x, results_df['F1-Score'], width, label='F1-Score', color='#4CAF50')
                ax.bar(x + width, results_df['Precision'], width, label='Precision', color='#FF9800')
                ax.set_xlabel('Models')
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Confusion matrix for best model
                if st.session_state.best_model:
                    st.markdown("### 🎯 Best Model Confusion Matrix")
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
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("Train models first")
    
    # TAB 6: EXPORT
    with tabs[5]:
        st.markdown("## Export Results")
        
        if st.session_state.best_model:
            if st.button("📄 Generate Report", type="primary"):
                with st.spinner("Generating..."):
                    try:
                        report_gen = ReportGenerator(st.session_state.config)
                        report_path = report_gen.generate_report(
                            profile=st.session_state.profile,
                            preprocessing_steps=st.session_state.pipeline,
                            results=st.session_state.training_results,
                            best_pipeline=st.session_state.best_model
                        )
                        
                        st.success(f"✅ Report: {report_path}")
                        
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                "⬇️ Download Report",
                                f,
                                file_name=Path(report_path).name,
                                mime='application/pdf',
                                type="primary"
                            )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Train models first")

if __name__ == "__main__":
    main()
