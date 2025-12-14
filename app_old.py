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
    ReportException, ErrorContext
)


# Page configuration
st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_THEME_CSS = """
<style>
    :root {
        --bg: #f4f6fb;
        --card: #ffffff;
        --border: rgba(15, 23, 42, 0.08);
        --accent: #0f62fe;
        --text: #0f172a;
        --muted: #475569;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        background-color: var(--bg);
    }

    body {
        background-color: var(--bg);
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a, #1f2937);
        color: #f8fafc;
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
    }

    .hero-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.3rem;
        font-size: 0.8rem;
        color: #a5b4fc;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .hero-subtitle {
        color: #d1d5db;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    .hero-metrics {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }

    .hero-metric {
        flex: 1;
        min-width: 120px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        display: block;
    }

    .metric-label {
        color: #cbd5f5;
        font-size: 0.9rem;
        letter-spacing: 0.04em;
    }

    .section-title {
        margin-bottom: 1rem;
        color: var(--text);
        font-weight: 600;
    }

    .status-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.9rem;
        background: #eef2ff;
        color: #3730a3;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: var(--text);
    }

    .stApp, .stApp .block-container {
        color: var(--text) !important;
    }

    .stApp .section-card,
    .stApp .grid-card,
    .stApp .stMarkdown,
    .stApp .stText,
    .stApp .stMetricValue,
    .stApp .stMetricLabel {
        color: var(--text) !important;
    }

    .stApp .stAlert,
    .stAlert p,
    .stAlert span,
    .stApp .stSelectbox>div>div,
    .stApp .stMultiSelect>div>div,
    .stApp .stRadio>div>label {
        color: var(--text) !important;
    }

    .stApp .stButton button {
        color: #ffffff !important;
    }

    .hero-card,
    .hero-card * {
        color: #f8fafc !important;
    }

    .section-card {
        background: #ffffff;
        border-color: rgba(15, 23, 42, 0.15);
    }

    .grid-card {
        background: #ffffff;
        border-color: rgba(15, 23, 42, 0.15);
    }

    .status-card {
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.12);
        background: #ffffff;
        min-height: 130px;
    }

    .message-card {
        background: #e2e8f0;
        color: #0f172a;
        box-shadow: none;
        border: 1px solid rgba(15, 23, 42, 0.1);
    }

    .message-card.warning {
        background: #fef9c3;
        color: #92400e;
        border-color: rgba(217, 119, 6, 0.2);
    }

    .message-card.error {
        background: #fee2e2;
        color: #991b1b;
        border-color: rgba(220, 38, 38, 0.2);
    }
</style>
"""


def apply_theme():
    st.markdown(APP_THEME_CSS, unsafe_allow_html=True)


def format_stats(value, default="—"):
    return str(value) if value not in (None, "") else default


def format_memory(df: pd.DataFrame):
    if df is None:
        return "—"
    try:
        mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        return f"{mb:.1f} MB"
    except Exception:
        return "—"


def render_hero_section():
    df = st.session_state.df
    rows = format_stats(len(df) if df is not None else None)
    cols = format_stats(df.shape[1] if df is not None else None)
    memory = format_memory(df)
    st.markdown(f"""
    <div class='hero-card'>
        <div>
            <p class='hero-eyebrow'>Enterprise-ready AutoML</p>
            <h1 class='hero-title'>Thoughtful automation, confident results</h1>
            <p class='hero-subtitle'>A quiet, professional workspace that keeps every step accountable—from ingestion
            to deployment—while keeping resource usage predictable on any laptop.</p>
        </div>
        <div class='hero-metrics'>
            <div class='hero-metric'>
                <span class='metric-value'>{rows}</span>
                <span class='metric-label'>Dataset rows</span>
            </div>
            <div class='hero-metric'>
                <span class='metric-value'>{cols}</span>
                <span class='metric-label'>Features</span>
            </div>
            <div class='hero-metric'>
                <span class='metric-value'>{memory}</span>
                <span class='metric-label'>Memory footprint</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_tiles():
    df_ready = st.session_state.df is not None
    pipeline_ready = st.session_state.preprocessing_pipeline is not None
    model_ready = st.session_state.best_model is not None

    statuses = [
        {
            'label': 'Dataset ingestion',
            'ready': df_ready,
            'detail': 'Upload and validate your CSV',
        },
        {
            'label': 'Pipeline readiness',
            'ready': pipeline_ready,
            'detail': 'Preprocessing configured and fitted',
        },
        {
            'label': 'Model training',
            'ready': model_ready,
            'detail': 'Train to unlock evaluation & export',
        },
    ]

    cols = st.columns(3, gap="large")
    for col, status in zip(cols, statuses):
        with col:
            badge = "Ready" if status['ready'] else "Pending"
            state_color = "status-tag"
            st.markdown(f"""
            <div class='grid-card status-card'>
                <div class='section-title'>{status['label']}</div>
                <span class='{state_color}'>{badge}</span>
                <p class='metric-label' style='margin-top:0.35rem;'>{status['detail']}</p>
            </div>
            """, unsafe_allow_html=True)


def render_message_panel():
    messages = st.session_state.messages[-12:]
    body = ""

    if not messages:
        body = "<div class='message-card info'>No updates yet. Upload data to get started.</div>"
    else:
        for msg in messages:
            level = "info"
            if 'ERROR' in msg or 'CRITICAL' in msg:
                level = "error"
            elif 'WARNING' in msg:
                level = "warning"

            safe_msg = msg.replace('<', '&lt;').replace('>', '&gt;')
            body += f"<div class='message-card {level}'>{safe_msg}</div>"

    panel_html = f"<div class='section-card'><div class='panel-title'>Live status & insights</div>{body}</div>"
    st.markdown(panel_html, unsafe_allow_html=True)


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
    apply_theme()
    render_hero_section()
    render_status_tiles()

    fast_only = False

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
    st.subheader("About this workspace")
    st.caption("A sleek, resource-conscious AutoML shell for analysts and developers alike.")

    with st.container():
        col_main, col_panel = st.columns([3, 1], gap="large")

        with col_main:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📁 Data Upload",
                "🔍 Data Profiling",
                "⚙️ Preprocessing",
                "🎯 Model Training",
                "📊 Results & Export"
            ])

            with tab1:
                st.subheader("📁 Upload Your Dataset")
                
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type=['csv', 'txt', 'tsv'],
                    help="Upload a CSV file (max 500MB)"
                )
                
                if uploaded_file is not None:
                    try:
                        if uploaded_file.size == 0:
                            st.error("❌ Uploaded file is empty")
                            st.session_state.messages.append("ERROR: Uploaded file is empty (0 bytes)")
                        elif uploaded_file.size > 500 * 1024 * 1024:
                            st.error("❌ File is too large (max 500MB)")
                            st.session_state.messages.append(f"ERROR: File too large ({uploaded_file.size / (1024**2):.1f}MB)")
                        else:
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
                                        st.success("✅ Data loaded successfully!")
                                        
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
                                        st.error("❌ Failed to load data. Please review the log panel on the right.")
                            
                            except IngestException as e:
                                st.error(f"❌ Data Ingestion Error: {e.message}")
                                st.session_state.messages.append(f"ERROR: {e.message}")
                            except Exception as e:
                                st.error(f"❌ Critical error: {str(e)[:200]}")
                                st.session_state.messages.append(f"CRITICAL: {str(e)[:200]}")
                                logging.error(f"Upload failed: {traceback.format_exc()}")
                            
                            finally:
                                try:
                                    if 'tmp_path' in locals():
                                        os.unlink(tmp_path)
                                except Exception as cleanup_error:
                                    logging.warning(f"Failed to clean up temp file: {cleanup_error}")
                    
                    except Exception as e:
                        st.error(f"❌ File upload validation failed: {str(e)[:200]}")
                        st.session_state.messages.append(f"ERROR: File validation failed: {str(e)[:200]}")

            with tab2:
                st.subheader("🔍 Data Quality Analysis")
                
                if st.session_state.df is None:
                    st.warning("⚠️ Please upload data first (Tab 1)")
                else:
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
                    
                    if st.session_state.profile:
                        profile = st.session_state.profile
                        
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

            with tab3:
                st.subheader("⚙️ Automated Preprocessing")
                
                if st.session_state.profile is None:
                    st.warning("⚠️ Please complete data profiling first (Tab 2)")
                else:
                    if st.button("🔧 Build & Apply Preprocessing Pipeline", type="primary"):
                        with st.spinner("Building preprocessing pipeline..."):
                            try:
                                target_col = st.session_state.profile['target_column']
                                
                                pipeline_builder = PreprocessingPipelineBuilder(st.session_state.config)
                                X, y, warnings = pipeline_builder.prepare_data(st.session_state.df, target_col)
                                
                                st.session_state.messages.extend(warnings)
                                
                                pipeline = pipeline_builder.build_pipeline(X, target_col, st.session_state.profile)
                                st.session_state.messages.extend(pipeline_builder.warnings)
                                
                                X_processed = pipeline_builder.fit_transform(X, y)
                                
                                st.session_state.preprocessing_pipeline = pipeline
                                st.session_state.X_processed = X_processed
                                st.session_state.y = y
                                
                                st.success("✅ Preprocessing complete!")
                                
                                st.subheader("Applied Preprocessing Steps")
                                for i, step in enumerate(pipeline_builder.preprocessing_steps, 1):
                                    st.write(f"{i}. {step}")
                                
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

            with tab4:
                st.subheader("🎯 Model Training & Selection")
                
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

            with tab5:
                st.subheader("📊 Results & Export")
                
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

        with col_panel:
            render_message_panel()


if __name__ == "__main__":
    main()
