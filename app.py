"""
Streamlit UI for AutoML System – Professional, Modern Design
Clean hierarchy, refined typography, and comprehensive error handling
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

from automl.config.settings import AutoMLConfig
from automl.data.ingestion import DataIngestor
from automl.data.profiling import DataProfiler
from automl.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from automl.models.trainer import ModelTrainer
from automl.reports.report_generator import ReportGenerator

from automl.utils.error_handlers import (
    IngestException, ProfilingException, PreprocessingException, TrainingException,
    ReportException, ErrorContext
)

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== THEME & STYLING ====================
THEME_CSS = """
<style>
    :root {
        --primary: #0f172a;
        --secondary: #1e293b;
        --accent: #3b82f6;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --bg: #f8fafc;
        --card: #ffffff;
        --border: #e2e8f0;
        --text: #0f172a;
        --muted: #64748b;
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }

    .stApp {
        background-color: var(--bg);
    }

    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    p, span, label {
        color: #000000;
    }

    .stMarkdown {
        color: #000000;
    }

    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-radius: 16px;
        padding: 3rem;
        margin-bottom: 2rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #000000;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
        max-width: 600px;
    }

    .hero-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }

    .stat {
        background-color: #f8f9fa;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        display: block;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #1a1a1a;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* Card Styles */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.75rem;
    }

    .card-subtitle {
        font-size: 0.95rem;
        color: #000000;
        line-height: 1.5;
    }

    /* Status Cards */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .status-card {
        background: var(--card);
        border: 2px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .status-card.ready {
        border-color: var(--success);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%);
    }

    .status-card.pending {
        border-color: var(--warning);
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, transparent 100%);
    }

    .status-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.5rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: capitalize;
    }

    .status-badge.ready {
        background: var(--success);
        color: white;
    }

    .status-badge.pending {
        background: var(--warning);
        color: white;
    }

    .status-hint {
        font-size: 0.85rem;
        color: #000000;
        margin-top: 0.5rem;
    }

    /* Messages */
    .message-panel {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    .message-panel-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .message {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .message.info {
        background: #eff6ff;
        color: #0c4a6e;
        border-left: 3px solid var(--accent);
    }

    .message.success {
        background: #f0fdf4;
        color: #15803d;
        border-left: 3px solid var(--success);
    }

    .message.warning {
        background: #fffbeb;
        color: #92400e;
        border-left: 3px solid var(--warning);
    }

    .message.error {
        background: #fef2f2;
        color: #7f1d1d;
        border-left: 3px solid var(--error);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--accent) 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: none;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }

    .stButton button:active {
        transform: translateY(0);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: none;
        color: #000000;
        padding: 1rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #000000;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent);
        border-bottom: 3px solid var(--accent);
        box-shadow: 0 3px 0 0 var(--accent) inset;
    }

    /* Forms */
    .stSelectbox, .stTextInput, .stTextArea {
        border-radius: 8px;
    }

    .stSelectbox>div>div, .stTextInput>div>div, .stTextArea>div>textarea {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: white;
        color: #000000;
    }

    .stSelectbox>div>div:focus-within, .stTextInput>div>div:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .stFileUploader {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.02) 0%, transparent 100%);
    }

    .stFileUploader:hover {
        border-color: var(--accent);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%);
    }

    /* Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border-right: 1px solid var(--border);
    }

    .stSidebar .stMarkdown {
        color: var(--text);
    }

    .stSidebar h3 {
        color: var(--primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }

    .sidebar-section {
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-section:last-child {
        border-bottom: none;
    }

    .sidebar-section > strong {
        display: block;
        color: var(--primary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* Metrics */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .stMetric {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.02) 0%, transparent 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid var(--border);
    }

    .stMetricValue {
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
        display: block;
    }

    .stMetricLabel {
        font-size: 0.85rem;
        color: #000000;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Data preview */
    .dataframe-container {
        background: var(--card);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    /* Alerts */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .stAlert > div > div > div {
        color: var(--text);
    }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session_state():
    defaults = {
        'config': AutoMLConfig(),
        'df': None,
        'profile': None,
        'preprocessing_pipeline': None,
        'X_processed': None,
        'y': None,
        'training_results': None,
        'best_model': None,
        'messages': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== UTILITIES ====================
def format_size(value, default="—"):
    return str(value) if value not in (None, "") else default

def format_memory(df):
    if df is None:
        return "—"
    try:
        return f"{df.memory_usage(deep=True).sum() / (1024**2):.1f} MB"
    except:
        return "—"

def log_message(msg, level="info"):
    st.session_state.messages.append({"text": msg, "level": level})

def render_messages():
    if not st.session_state.messages:
        return

    with st.container():
        st.markdown('<div class="message-panel">', unsafe_allow_html=True)
        st.markdown('<div class="message-panel-title">📋 Status & Insights</div>', unsafe_allow_html=True)

        for msg_obj in st.session_state.messages[-8:]:
            msg = msg_obj["text"]
            level = msg_obj.get("level", "info")
            st.markdown(f'<div class="message {level}">{msg}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

def render_status_overview():
    df_ready = st.session_state.df is not None
    pipeline_ready = st.session_state.preprocessing_pipeline is not None
    model_ready = st.session_state.best_model is not None

    statuses = [
        {"label": "Data Upload", "ready": df_ready, "hint": "CSV file loaded and validated"},
        {"label": "Preprocessing", "ready": pipeline_ready, "hint": "Pipeline built and ready"},
        {"label": "Model Training", "ready": model_ready, "hint": "Models trained and exported"},
    ]

    st.markdown('<div class="status-grid">', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")

    for col, status in zip(cols, statuses):
        with col:
            state = "ready" if status["ready"] else "pending"
            badge_text = "✓ Ready" if status["ready"] else "○ Pending"
            
            st.markdown(f"""
            <div class='status-card {state}'>
                <div class='status-label'>{status['label']}</div>
                <span class='status-badge {state}'>{badge_text}</span>
                <div class='status-hint'>{status['hint']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_hero():
    df = st.session_state.df
    rows = format_size(len(df) if df is not None else None)
    cols = format_size(df.shape[1] if df is not None else None)
    memory = format_memory(df)

    st.markdown(f"""
    <div class='hero'>
        <h1>AutoML Workspace</h1>
        <p>Build, train, and deploy machine learning models with a single pipeline. No configuration required.</p>
        <div class='hero-stats'>
            <div class='stat'>
                <span class='stat-value'>{rows}</span>
                <span class='stat-label'>Rows</span>
            </div>
            <div class='stat'>
                <span class='stat-value'>{cols}</span>
                <span class='stat-label'>Features</span>
            </div>
            <div class='stat'>
                <span class='stat-value'>{memory}</span>
                <span class='stat-label'>Memory</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**Training**")
            fast_only = st.checkbox("Fast models only", value=False, help="Skip slow models for quick prototyping")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**Feature Engineering**")
            st.session_state.config.ENABLE_AUTO_FEATURE_ENGINEERING = st.checkbox(
                "Auto features",
                value=True,
                help="Polynomial & interaction features"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**About**")
            st.caption("Professional AutoML for everyone. Quiet, predictable, reproducible.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Header
    render_hero()
    render_status_overview()

    # Two-column layout
    col_main, col_panel = st.columns([3, 1], gap="large")

    with col_panel:
        render_messages()

    with col_main:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Upload",
            "🔍 Profile",
            "⚙️ Prepare",
            "🎯 Train",
            "📊 Export"
        ])

        # TAB 1: UPLOAD
        with tab1:
            st.markdown("### Upload Dataset")
            st.markdown("Drag and drop a CSV file, or click to select one.")

            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt', 'tsv'], label_visibility="hidden")

            if uploaded_file:
                try:
                    if uploaded_file.size == 0:
                        st.error("File is empty")
                        log_message("ERROR: Empty file uploaded", "error")
                    elif uploaded_file.size > 500 * 1024 * 1024:
                        st.error("File too large (max 500MB)")
                        log_message(f"ERROR: File {uploaded_file.size / (1024**2):.1f}MB exceeds limit", "error")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        try:
                            with st.spinner("Loading dataset..."):
                                ingestor = DataIngestor(st.session_state.config)
                                df, messages = ingestor.ingest(tmp_path)
                                st.session_state.messages = [{"text": m, "level": "info"} for m in messages]

                                if df is not None:
                                    st.session_state.df = df
                                    st.success("✓ Dataset loaded successfully!")
                                    log_message(f"Loaded {len(df):,} rows × {df.shape[1]} columns", "success")

                                    st.markdown("### Preview")
                                    st.dataframe(df.head(100), use_container_width=True)

                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.metric("Rows", f"{len(df):,}")
                                    with cols[1]:
                                        st.metric("Columns", df.shape[1])
                                    with cols[2]:
                                        st.metric("Memory", f"{format_memory(df)}")
                                else:
                                    st.error("Failed to load dataset")
                                    log_message("ERROR: Data ingestion failed", "error")
                        except Exception as e:
                            st.error(f"Error: {str(e)[:100]}")
                            log_message(f"ERROR: {str(e)[:100]}", "error")
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                except Exception as e:
                    st.error(f"File error: {str(e)[:100]}")
                    log_message(f"ERROR: {str(e)[:100]}", "error")

        # TAB 2: PROFILE
        with tab2:
            st.markdown("### Data Quality Analysis")

            if st.session_state.df is None:
                st.info("Upload a dataset first (Tab: Upload)")
            else:
                target_col = st.selectbox(
                    "Select target column",
                    options=st.session_state.df.columns.tolist(),
                    help="Column to predict"
                )

                if st.button("Analyze Dataset", type="primary"):
                    with st.spinner("Analyzing..."):
                        profiler = DataProfiler(st.session_state.config)
                        profile = profiler.profile_dataset(st.session_state.df, target_col)

                        st.session_state.profile = profile
                        st.session_state.profile['target_column'] = target_col
                        st.session_state.messages = [{"text": m, "level": "info"} for m in profiler.warnings]

                        st.success("✓ Analysis complete!")

                if st.session_state.profile:
                    profile = st.session_state.profile
                    basic = profile.get('basic', {})

                    st.markdown("### Dataset Overview")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Rows", f"{basic.get('n_rows', 0):,}")
                    with cols[1]:
                        st.metric("Columns", basic.get('n_cols', 0))
                    with cols[2]:
                        st.metric("Memory", f"{basic.get('memory_mb', 0):.1f} MB")
                    with cols[3]:
                        missing = profile.get('missing', {})
                        st.metric("Missing", f"{missing.get('total_missing', 0):,}")

                    if 'target' in profile:
                        target_info = profile['target']
                        st.markdown("### Target Variable")
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Type", target_info.get('task_type', '?').title())
                        with cols[1]:
                            st.metric("Unique", target_info.get('n_unique', 0))

                        if 'class_distribution' in target_info:
                            st.bar_chart(pd.Series(target_info['class_distribution']))

                    warnings = profile.get('warnings', [])
                    if warnings:
                        st.markdown("### Quality Issues")
                        for w in warnings[:10]:
                            if 'CRITICAL' in w or 'ERROR' in w:
                                st.error(w)
                            elif 'WARNING' in w:
                                st.warning(w)
                            else:
                                st.info(w)

        # TAB 3: PREPARE
        with tab3:
            st.markdown("### Preprocessing")

            if st.session_state.profile is None:
                st.info("Complete profiling first (Tab: Profile)")
            else:
                if st.button("Build Pipeline", type="primary"):
                    with st.spinner("Building..."):
                        try:
                            target_col = st.session_state.profile['target_column']
                            builder = PreprocessingPipelineBuilder(st.session_state.config)
                            X, y, warnings = builder.prepare_data(st.session_state.df, target_col)
                            st.session_state.messages.extend([{"text": w, "level": "info"} for w in warnings])

                            pipeline = builder.build_pipeline(X, target_col, st.session_state.profile)
                            st.session_state.messages.extend([{"text": w, "level": "info"} for w in builder.warnings])

                            X_processed = builder.fit_transform(X, y)

                            st.session_state.preprocessing_pipeline = pipeline
                            st.session_state.X_processed = X_processed
                            st.session_state.y = y

                            st.success("✓ Pipeline ready!")
                            log_message(f"Pipeline: {len(X.columns)} → {X_processed.shape[1]} features", "success")

                            st.markdown("### Steps Applied")
                            for i, step in enumerate(builder.preprocessing_steps, 1):
                                st.caption(f"{i}. {step}")

                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("Input Features", len(X.columns))
                            with cols[1]:
                                st.metric("Output Features", X_processed.shape[1])

                        except PreprocessingException as e:
                            st.error(f"Preprocessing failed: {e.message}")
                            log_message(f"ERROR: {e.message}", "error")
                        except Exception as e:
                            st.error(f"Error: {str(e)[:100]}")
                            log_message(f"ERROR: {str(e)[:100]}", "error")

        # TAB 4: TRAIN
        with tab4:
            st.markdown("### Model Training")

            if st.session_state.X_processed is None:
                st.info("Complete preprocessing first (Tab: Prepare)")
            else:
                if st.button("Train Models", type="primary"):
                    with st.spinner("Training..."):
                        try:
                            trainer = ModelTrainer(st.session_state.config)
                            results = trainer.train_models(
                                st.session_state.X_processed,
                                st.session_state.y,
                                task_type='auto',
                                fast_only=fast_only
                            )

                            st.session_state.training_results = results
                            st.session_state.best_model = trainer.best_model
                            st.session_state.messages.extend([{"text": w, "level": "info"} for w in trainer.warnings])

                            st.success("✓ Training complete!")

                        except Exception as e:
                            st.error(f"Training failed: {str(e)[:100]}")
                            log_message(f"ERROR: {str(e)[:100]}", "error")

                if st.session_state.training_results:
                    results = st.session_state.training_results

                    st.markdown("### Best Model")
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Model", results.get('best_model_name', 'N/A'))
                    with cols[1]:
                        st.metric("Score", f"{results.get('best_score', 0):.4f}")

                    all_results = results.get('all_results', [])
                    if all_results:
                        st.markdown("### All Models")
                        results_data = []
                        for r in all_results:
                            if r.get('status') == 'success':
                                row = {
                                    'Model': r['model_name'],
                                    'Status': '✓',
                                    'Time (s)': r.get('training_time', 0),
                                }
                                metrics = r.get('metrics', {})
                                for k, v in metrics.items():
                                    if v is not None:
                                        row[k.replace('_', ' ').title()] = v
                                results_data.append(row)
                            else:
                                results_data.append({
                                    'Model': r['model_name'],
                                    'Status': '✗',
                                    'Error': r.get('error', '?')[:50]
                                })

                        st.dataframe(pd.DataFrame(results_data), use_container_width=True)

        # TAB 5: EXPORT
        with tab5:
            st.markdown("### Export & Deploy")

            if st.session_state.best_model is None:
                st.info("Train models first (Tab: Train)")
            else:
                st.success("✓ Model ready!")

                cols = st.columns(2)

                with cols[0]:
                    st.markdown("#### Save Model")
                    if st.button("Export (.pkl)", type="primary"):
                        try:
                            import joblib
                            Path("outputs").mkdir(exist_ok=True)
                            model_path = "outputs/best_model.pkl"

                            joblib.dump({
                                'pipeline': st.session_state.preprocessing_pipeline,
                                'model': st.session_state.best_model,
                                'config': st.session_state.config,
                            }, model_path)

                            st.success(f"✓ Saved to {model_path}")

                            with open(model_path, 'rb') as f:
                                st.download_button(
                                    "Download Model",
                                    data=f,
                                    file_name="automl_model.pkl",
                                    mime="application/octet-stream"
                                )
                        except Exception as e:
                            st.error(f"Export failed: {str(e)[:100]}")

                with cols[1]:
                    st.markdown("#### Generate Report")
                    if st.button("Export Report", type="primary"):
                        try:
                            Path("outputs").mkdir(exist_ok=True)
                            report_path = "outputs/report.html"

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

                            st.success(f"✓ Report saved: {report_path}")

                            with open(report_path, 'r', encoding='utf-8') as f:
                                st.download_button(
                                    "Download Report",
                                    data=f,
                                    file_name="automl_report.html",
                                    mime="text/html"
                                )
                        except Exception as e:
                            st.error(f"Report failed: {str(e)[:100]}")

                st.markdown("#### Usage Example")
                st.code("""
import joblib
import pandas as pd

model_data = joblib.load('automl_model.pkl')
X_new = pd.read_csv('new_data.csv')
X_processed = model_data['pipeline'].transform(X_new)
predictions = model_data['model'].predict(X_processed)
print(predictions)
""", language='python')

if __name__ == "__main__":
    main()
