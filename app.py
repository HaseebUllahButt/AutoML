"""
Streamlit UI for AutoML System
User-friendly interface with comprehensive error handling and state management
Enhanced with detailed error tracking, validation, and recovery mechanisms
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import time
import logging
import traceback

# Import our AutoML components
from automl.config.settings import AutoMLConfig
from automl.data.ingestion import DataIngestor
from automl.data.profiling import DataProfiler
from automl.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from automl.models.trainer import ModelTrainer
from automl.reports.report_generator import ReportGenerator

# Import error handling utilities
from automl.utils.error_handlers import (
    IngestException, ProfilingException, PreprocessingException, TrainingException,
    ReportException, ErrorContext, ErrorCollector
)


# Page configuration
st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e7f3ff;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = AutoMLConfig()
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    
    if 'preprocessing_pipeline' not in st.session_state:
        st.session_state.preprocessing_pipeline = None
    
    if 'X_processed' not in st.session_state:
        st.session_state.X_processed = None
    
    if 'y' not in st.session_state:
        st.session_state.y = None
    
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def display_messages():
    """Display collected messages with enhanced formatting"""
    if st.session_state.messages:
        for msg in st.session_state.messages[-20:]:  # Show last 20 messages
            try:
                if 'ERROR' in msg or 'CRITICAL' in msg:
                    st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
                elif 'WARNING' in msg:
                    st.markdown(f'<div class="warning-box">⚠️ {msg}</div>', unsafe_allow_html=True)
                elif 'INFO' in msg or '✓' in msg:
                    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)
                else:
                    st.info(msg)
            except Exception as e:
                # Fallback if markdown rendering fails
                st.write(msg)


def log_exception_details(exc, severity: str = "ERROR"):
    """Record exception context for debugging."""
    context = getattr(exc, 'context', None)
    if context:
        context_str = repr(context)
        st.session_state.messages.append(f"{severity}: Details -> {context_str[:500]}")
        logging.error("Exception context (%s): %s", exc.__class__.__name__, context_str)


def safe_ui_operation(operation_name: str, operation_func, *args, **kwargs):
    """
    Safely execute a UI operation with comprehensive error handling
    
    Args:
        operation_name: Name of the operation for error tracking
        operation_func: Function to execute
        args, kwargs: Arguments for the function
    
    Returns:
        Tuple of (success: bool, result: Any, error_message: str)
    """
    try:
        with ErrorContext(operation_name):
            result = operation_func(*args, **kwargs)
        return True, result, None
    
    except IngestException as e:
        error_msg = f"Data Ingestion Error: {e.message}"
        st.session_state.messages.append(f"ERROR: {error_msg}")
        log_exception_details(e)
        return False, None, error_msg
    
    except ProfilingException as e:
        error_msg = f"Profiling Error: {e.message}"
        st.session_state.messages.append(f"ERROR: {error_msg}")
        log_exception_details(e)
        return False, None, error_msg
    
    except PreprocessingException as e:
        error_msg = f"Preprocessing Error: {e.message}"
        st.session_state.messages.append(f"ERROR: {error_msg}")
        log_exception_details(e)
        return False, None, error_msg
    
    except TrainingException as e:
        error_msg = f"Training Error: {e.message}"
        st.session_state.messages.append(f"ERROR: {error_msg}")
        log_exception_details(e)
        return False, None, error_msg
    
    except ReportException as e:
        error_msg = f"Report Generation Error: {e.message}"
        st.session_state.messages.append(f"ERROR: {error_msg}")
        log_exception_details(e)
        return False, None, error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error in {operation_name}: {str(e)[:200]}"
        st.session_state.messages.append(f"CRITICAL: {error_msg}")
        logging.error("%s failed: %s", operation_name, traceback.format_exc())
        return False, None, error_msg


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 AutoML System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Training Options")
        fast_only = st.checkbox("Fast models only", value=False, help="Train only fast models for quick testing")
        
        st.subheader("Pipeline Steps")
        st.session_state.config.ENABLE_AUTO_FEATURE_ENGINEERING = st.checkbox(
            "Auto feature engineering", 
            value=True,
            help="Automatically create polynomial and interaction features"
        )
        
        st.markdown("---")
        st.subheader("About")
        st.info("""
        **AutoML System v1.0**
        
        Fully automated machine learning pipeline that handles:
        - Data ingestion & validation
        - Quality analysis
        - Preprocessing & cleaning
        - Model selection & training
        - Report generation
        
        Handles 50+ edge cases!
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data Upload",
        "🔍 Data Profiling",
        "⚙️ Preprocessing",
        "🎯 Model Training",
        "📊 Results & Export"
    ])
    
    # ==================== TAB 1: Data Upload ====================
    with tab1:
        st.header("📁 Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv', 'txt', 'tsv'],
            help="Upload a CSV file (max 500MB)"
        )
        
        if uploaded_file is not None:
            # Validate file before saving
            try:
                if uploaded_file.size == 0:
                    st.error("❌ Uploaded file is empty")
                    st.session_state.messages.append("ERROR: Uploaded file is empty (0 bytes)")
                elif uploaded_file.size > 500 * 1024 * 1024:  # 500MB
                    st.error("❌ File is too large (max 500MB)")
                    st.session_state.messages.append(f"ERROR: File too large ({uploaded_file.size / (1024**2):.1f}MB)")
                else:
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        with st.spinner("🔄 Ingesting data..."):
                            ingestor = DataIngestor(st.session_state.config)
                            df, messages = ingestor.ingest(tmp_path)
                            
                            st.session_state.messages = messages
                            
                            if df is not None:
                                st.session_state.df = df
                                st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                                
                                # Display preview
                                st.subheader("Data Preview")
                                st.dataframe(df.head(100), use_container_width=True)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Rows", f"{len(df):,}")
                                with col2:
                                    st.metric("Columns", len(df.columns))
                                with col3:
                                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
                            else:
                                st.markdown('<div class="error-box">❌ Failed to load data. Check messages below.</div>', unsafe_allow_html=True)
                    
                    except IngestException as e:
                        st.error(f"❌ Data Ingestion Error: {e.message}")
                        st.session_state.messages.append(f"ERROR: {e.message}")
                    except Exception as e:
                        st.error(f"❌ Critical error: {str(e)[:200]}")
                        st.session_state.messages.append(f"CRITICAL: {str(e)[:200]}")
                        logging.error(f"Upload failed: {traceback.format_exc()}")
                    
                    finally:
                        # Clean up temp file
                        try:
                            if 'tmp_path' in locals():
                                os.unlink(tmp_path)
                        except Exception as cleanup_error:
                            logging.warning(f"Failed to clean up temp file: {cleanup_error}")
            
            except Exception as e:
                st.error(f"❌ File upload validation failed: {str(e)[:200]}")
                st.session_state.messages.append(f"ERROR: File validation failed: {str(e)[:200]}")
        
        # Display messages
        if st.session_state.messages:
            with st.expander("📋 Ingestion Messages", expanded=False):
                display_messages()
    
    # ==================== TAB 2: Data Profiling ====================
    with tab2:
        st.header("🔍 Data Quality Analysis")
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first (Tab 1)")
        else:
            # Select target column
            st.subheader("Select Target Column")
            target_col = st.selectbox(
                "Target column for prediction:",
                options=st.session_state.df.columns.tolist(),
                help="The column you want to predict"
            )
            
            if st.button("🔬 Analyze Data Quality", type="primary"):
                with st.spinner("Analyzing data quality..."):
                    profiler = DataProfiler(st.session_state.config)
                    profile = profiler.profile_dataset(st.session_state.df, target_col)
                    
                    st.session_state.profile = profile
                    st.session_state.profile['target_column'] = target_col
                    st.session_state.messages.extend(profiler.warnings)
                    st.session_state.messages.extend(profiler.recommendations)
                    
                    st.success("✅ Profiling complete!")
            
            # Display profile results
            if st.session_state.profile:
                profile = st.session_state.profile
                
                # Basic info
                st.subheader("📊 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                basic = profile.get('basic', {})
                with col1:
                    st.metric("Rows", f"{basic.get('n_rows', 0):,}")
                with col2:
                    st.metric("Columns", basic.get('n_cols', 0))
                with col3:
                    st.metric("Memory", f"{basic.get('memory_mb', 0):.1f} MB")
                with col4:
                    missing = profile.get('missing', {})
                    st.metric("Missing Values", f"{missing.get('total_missing', 0):,}")
                
                # Target info
                if 'target' in profile:
                    st.subheader("🎯 Target Variable Analysis")
                    target_info = profile['target']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Task Type", target_info.get('task_type', 'Unknown').title())
                    with col2:
                        st.metric("Unique Values", target_info.get('n_unique', 0))
                    
                    if 'class_distribution' in target_info:
                        st.write("**Class Distribution:**")
                        class_dist = target_info['class_distribution']
                        st.bar_chart(pd.Series(class_dist))
                
                # Data quality warnings
                warnings = profile.get('warnings', [])
                if warnings:
                    st.subheader("⚠️ Data Quality Issues")
                    for warning in warnings[:15]:
                        if 'CRITICAL' in warning or 'ERROR' in warning:
                            st.error(warning)
                        elif 'WARNING' in warning:
                            st.warning(warning)
                        else:
                            st.info(warning)
    
    # ==================== TAB 3: Preprocessing ====================
    with tab3:
        st.header("⚙️ Automated Preprocessing")
        
        if st.session_state.profile is None:
            st.warning("⚠️ Please complete data profiling first (Tab 2)")
        else:
            if st.button("🔧 Build & Apply Preprocessing Pipeline", type="primary"):
                with st.spinner("Building preprocessing pipeline..."):
                    try:
                        target_col = st.session_state.profile['target_column']
                        
                        # Prepare data
                        pipeline_builder = PreprocessingPipelineBuilder(st.session_state.config)
                        X, y, warnings = pipeline_builder.prepare_data(st.session_state.df, target_col)
                        
                        st.session_state.messages.extend(warnings)
                        
                        # Build pipeline
                        pipeline = pipeline_builder.build_pipeline(X, target_col, st.session_state.profile)
                        st.session_state.messages.extend(pipeline_builder.warnings)
                        
                        # Fit and transform
                        X_processed = pipeline_builder.fit_transform(X, y)
                        
                        st.session_state.preprocessing_pipeline = pipeline
                        st.session_state.X_processed = X_processed
                        st.session_state.y = y
                        
                        st.success("✅ Preprocessing complete!")
                        
                        # Show preprocessing steps
                        st.subheader("Applied Preprocessing Steps")
                        for i, step in enumerate(pipeline_builder.preprocessing_steps, 1):
                            st.write(f"{i}. {step}")
                        
                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Features", len(X.columns))
                        with col2:
                            st.metric("Processed Features", len(X_processed.columns) if hasattr(X_processed, 'columns') else X_processed.shape[1])
                        
                    except PreprocessingException as e:
                        st.error(f"Preprocessing failed: {e.message}")
                        log_exception_details(e)
                    except Exception as e:
                        error_text = f"Preprocessing failed: {str(e)[:200]}"
                        st.error(error_text)
                        st.session_state.messages.append(f"CRITICAL: {error_text}")
                        logging.error("Preprocessing pipeline unexpected error: %s", traceback.format_exc())
            
            # Display messages
            if st.session_state.messages:
                with st.expander("📋 Preprocessing Messages", expanded=False):
                    display_messages()
    
    # ==================== TAB 4: Model Training ====================
    with tab4:
        st.header("🎯 Model Training & Selection")
        
        if st.session_state.X_processed is None:
            st.warning("⚠️ Please complete preprocessing first (Tab 3)")
        else:
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training multiple models... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        trainer = ModelTrainer(st.session_state.config)
                        
                        status_text.text("Training models...")
                        results = trainer.train_models(
                            st.session_state.X_processed,
                            st.session_state.y,
                            task_type='auto',
                            fast_only=fast_only
                        )
                        
                        progress_bar.progress(100)
                        
                        st.session_state.training_results = results
                        st.session_state.best_model = trainer.best_model
                        st.session_state.messages.extend(trainer.warnings)
                        
                        st.success("✅ Training complete!")
                        
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.session_state.messages.append(f"ERROR: {e}")
                    
                    finally:
                        progress_bar.empty()
                        status_text.empty()
            
            # Display results
            if st.session_state.training_results:
                results = st.session_state.training_results
                
                if 'error' in results:
                    st.error(f"Training Error: {results['error']}")
                else:
                    st.subheader("🏆 Best Model")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model", results.get('best_model_name', 'N/A'))
                    with col2:
                        st.metric("Score", f"{results.get('best_score', 0):.4f}")
                    
                    st.subheader("📊 All Models Performance")
                    all_results = results.get('all_results', [])
                    
                    if all_results:
                        # Create results DataFrame
                        results_data = []
                        for r in all_results:
                            if r.get('status') == 'success':
                                row = {
                                    'Model': r['model_name'],
                                    'Status': '✓',
                                    'Time (s)': r.get('training_time', 0),
                                }
                                metrics = r.get('metrics', {})
                                for key, value in metrics.items():
                                    if value is not None:
                                        row[key.replace('_', ' ').title()] = value
                                results_data.append(row)
                            else:
                                results_data.append({
                                    'Model': r['model_name'],
                                    'Status': '✗',
                                    'Error': r.get('error', 'Unknown')[:100]
                                })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        st.warning("No models were trained.")
    
    # ==================== TAB 5: Results & Export ====================
    with tab5:
        st.header("📊 Results & Export")
        
        if st.session_state.best_model is None:
            st.warning("⚠️ Please train models first (Tab 4)")
        else:
            st.success("✅ Model is ready for export!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💾 Export Model")
                if st.button("Download Model (.pkl)", type="primary"):
                    try:
                        import joblib
                        model_path = "outputs/best_model.pkl"
                        Path("outputs").mkdir(exist_ok=True)
                        
                        joblib.dump({
                            'preprocessing_pipeline': st.session_state.preprocessing_pipeline,
                            'model': st.session_state.best_model,
                            'config': st.session_state.config,
                        }, model_path)
                        
                        st.success(f"✅ Model saved to {model_path}")
                        
                        with open(model_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Model File",
                                data=f,
                                file_name="automl_model.pkl",
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Failed to save model: {e}")
            
            with col2:
                st.subheader("📄 Generate Report")
                if st.button("Generate HTML Report", type="primary"):
                    try:
                        report_path = "outputs/automl_report.html"
                        Path("outputs").mkdir(exist_ok=True)
                        
                        generator = ReportGenerator(st.session_state.config)
                        
                        # Get preprocessing steps
                        if hasattr(st.session_state.preprocessing_pipeline, 'steps'):
                            steps = [name for name, _ in st.session_state.preprocessing_pipeline.steps]
                        else:
                            steps = ["Preprocessing applied"]
                        
                        generator.generate_report(
                            profile=st.session_state.profile,
                            preprocessing_steps=steps,
                            training_results=st.session_state.training_results,
                            target_col=st.session_state.profile.get('target_column', 'target'),
                            output_path=report_path
                        )
                        
                        st.success(f"✅ Report generated: {report_path}")
                        
                        with open(report_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="📥 Download HTML Report",
                                data=f,
                                file_name="automl_report.html",
                                mime="text/html"
                            )
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
            
            # Usage example
            st.subheader("💡 How to Use Your Model")
            st.code("""
import joblib
import pandas as pd

# Load the saved model
model_data = joblib.load('automl_model.pkl')
preprocessing_pipeline = model_data['preprocessing_pipeline']
model = model_data['model']

# Prepare new data (same format as training data, without target column)
new_data = pd.read_csv('new_data.csv')

# Preprocess
X_processed = preprocessing_pipeline.transform(new_data)

# Make predictions
predictions = model.predict(X_processed)

print(predictions)
""", language='python')


if __name__ == "__main__":
    main()
