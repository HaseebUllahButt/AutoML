"""
Report generation system
Creates comprehensive HTML reports with visualizations and explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..config.settings import AutoMLConfig


class ReportGenerator:
    """Generates comprehensive HTML reports"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        
    def generate_report(self, 
                       profile: Dict,
                       preprocessing_steps: List[str],
                       training_results: Dict,
                       target_col: str,
                       output_path: str = 'automl_report.html') -> str:
        """
        Generate a comprehensive HTML report
        
        Args:
            profile: Data profile from DataProfiler
            preprocessing_steps: List of preprocessing steps applied
            training_results: Results from ModelTrainer
            target_col: Name of target column
            output_path: Where to save the report
            
        Returns:
            Path to generated report
        """
        html = self._generate_html(profile, preprocessing_steps, training_results, target_col)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def _generate_html(self, profile, preprocessing_steps, training_results, target_col) -> str:
        """Generate the HTML content"""
        
        # Build sections
        sections = []
        
        # Header
        sections.append(self._html_header())
        
        # Executive Summary
        sections.append(self._html_executive_summary(profile, training_results, target_col))
        
        # Data Quality Analysis
        sections.append(self._html_data_quality(profile))
        
        # Preprocessing Steps
        sections.append(self._html_preprocessing(preprocessing_steps))
        
        # Model Training Results
        sections.append(self._html_model_results(training_results))
        
        # Recommendations
        sections.append(self._html_recommendations(profile, training_results))
        
        # Footer
        sections.append(self._html_footer())
        
        return '\n'.join(sections)
    
    def _html_header(self) -> str:
        """Generate HTML header"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .summary-box {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric-card .label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 3px;
        }
        
        .error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 3px;
        }
        
        .info {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 10px 0;
            border-radius: 3px;
        }
        
        .success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
            border-radius: 3px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background: #3498db;
            color: white;
            font-weight: 600;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .best-model {
            background: #d4edda !important;
        }
        
        ul {
            margin: 15px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 8px 0;
        }
        
        .timestamp {
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 40px;
        }
        
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        .code-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AutoML System Report</h1>
"""
    
    def _html_executive_summary(self, profile, training_results, target_col) -> str:
        """Generate executive summary"""
        basic = profile.get('basic', {})
        target_info = profile.get('target', {})
        task_type = training_results.get('task_type', 'Unknown')
        best_model = training_results.get('best_model_name', 'None')
        best_score = training_results.get('best_score', 0)
        
        return f"""
        <div class="summary-box">
            <h2>📊 Executive Summary</h2>
            <p><strong>Target Column:</strong> <code>{target_col}</code></p>
            <p><strong>Task Type:</strong> {task_type.title()}</p>
            <p><strong>Best Model:</strong> {best_model} (Score: {best_score})</p>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="value">{basic.get('n_rows', 0):,}</div>
                    <div class="label">Rows</div>
                </div>
                <div class="metric-card">
                    <div class="value">{basic.get('n_cols', 0)}</div>
                    <div class="label">Columns</div>
                </div>
                <div class="metric-card">
                    <div class="value">{basic.get('memory_mb', 0):.1f} MB</div>
                    <div class="label">Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="value">{target_info.get('n_unique', 0)}</div>
                    <div class="label">Target Classes/Values</div>
                </div>
            </div>
        </div>
"""
    
    def _html_data_quality(self, profile) -> str:
        """Generate data quality section"""
        missing_info = profile.get('missing', {})
        duplicates = profile.get('duplicates', {})
        warnings = profile.get('warnings', [])
        
        html = "<h2>🔍 Data Quality Analysis</h2>"
        
        # Missing values
        html += f"""
        <h3>Missing Values</h3>
        <p>Total missing values: <strong>{missing_info.get('total_missing', 0):,}</strong></p>
        """
        
        high_missing = missing_info.get('high_missing_columns', {})
        if high_missing:
            html += "<p>⚠️ Columns with >50% missing values:</p><ul>"
            for col, pct in list(high_missing.items())[:10]:
                html += f"<li><code>{col}</code>: {pct:.1f}%</li>"
            html += "</ul>"
        
        # Duplicates
        n_dups = duplicates.get('n_duplicates', 0)
        if n_dups > 0:
            html += f"""
        <h3>Duplicate Rows</h3>
        <div class="warning">
            Found {n_dups:,} duplicate rows ({duplicates.get('duplicate_ratio', 0)*100:.1f}%)
        </div>
"""
        
        # Warnings
        if warnings:
            html += "<h3>Data Quality Warnings</h3>"
            for warning in warnings[:20]:  # Limit to first 20
                if 'CRITICAL' in warning or 'ERROR' in warning:
                    css_class = 'error'
                elif 'WARNING' in warning:
                    css_class = 'warning'
                else:
                    css_class = 'info'
                html += f'<div class="{css_class}">{warning}</div>'
        
        return html
    
    def _html_preprocessing(self, preprocessing_steps) -> str:
        """Generate preprocessing section"""
        html = "<h2>⚙️ Preprocessing Pipeline</h2>"
        html += "<p>The following preprocessing steps were applied:</p><ol>"
        
        for step in preprocessing_steps:
            html += f"<li>{step}</li>"
        
        html += "</ol>"
        return html
    
    def _html_model_results(self, training_results) -> str:
        """Generate model results section"""
        html = "<h2>🎯 Model Training Results</h2>"
        
        all_results = training_results.get('all_results', [])
        task_type = training_results.get('task_type', 'classification')
        best_model = training_results.get('best_model_name', '')
        
        if not all_results:
            html += "<p>No models were trained successfully.</p>"
            return html
        
        # Results table
        html += "<table><thead><tr>"
        html += "<th>Model</th><th>Status</th><th>Training Time (s)</th>"
        
        if task_type == 'classification':
            html += "<th>Test Accuracy</th><th>Test F1</th><th>Test Precision</th><th>Test Recall</th>"
        else:
            html += "<th>Test RMSE</th><th>Test MAE</th><th>Test R²</th>"
        
        html += "</tr></thead><tbody>"
        
        for result in all_results:
            model_name = result.get('model_name', 'Unknown')
            status = result.get('status', 'unknown')
            
            row_class = ' class="best-model"' if model_name == best_model else ''
            html += f"<tr{row_class}>"
            html += f"<td><strong>{model_name}</strong></td>"
            
            if status == 'success':
                metrics = result.get('metrics', {})
                training_time = result.get('training_time', 0)
                
                html += f"<td>✓ Success</td><td>{training_time:.2f}</td>"
                
                if task_type == 'classification':
                    html += f"<td>{metrics.get('test_accuracy', 'N/A')}</td>"
                    html += f"<td>{metrics.get('test_f1', 'N/A')}</td>"
                    html += f"<td>{metrics.get('test_precision', 'N/A')}</td>"
                    html += f"<td>{metrics.get('test_recall', 'N/A')}</td>"
                else:
                    html += f"<td>{metrics.get('test_rmse', 'N/A')}</td>"
                    html += f"<td>{metrics.get('test_mae', 'N/A')}</td>"
                    html += f"<td>{metrics.get('test_r2', 'N/A')}</td>"
            else:
                error = result.get('error', 'Unknown error')
                html += f'<td>✗ Failed</td><td colspan="5"><small>{error[:100]}</small></td>'
            
            html += "</tr>"
        
        html += "</tbody></table>"
        
        # Best model highlight
        best_score = training_results.get('best_score', 0)
        html += f"""
        <div class="success">
            <strong>🏆 Best Model:</strong> {best_model} with score {best_score}
        </div>
"""
        
        return html
    
    def _html_recommendations(self, profile, training_results) -> str:
        """Generate recommendations section"""
        html = "<h2>💡 Recommendations & Next Steps</h2>"
        
        recommendations = profile.get('recommendations', [])
        
        if recommendations:
            html += "<h3>Data Quality Improvements</h3><ul>"
            for rec in recommendations[:10]:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        # Model-specific recommendations
        html += "<h3>Model Improvement Suggestions</h3><ul>"
        html += "<li>Try feature engineering to create domain-specific features</li>"
        html += "<li>Collect more data if possible (especially for minority classes)</li>"
        html += "<li>Consider ensemble methods for better performance</li>"
        html += "<li>Perform more extensive hyperparameter tuning</li>"
        html += "<li>Check for data leakage in features</li>"
        html += "</ul>"
        
        # Usage instructions
        html += """
        <h3>How to Use This Model</h3>
        <div class="code-block">
            <pre><code>import joblib

# Load the model
model = joblib.load('best_model.pkl')

# Make predictions
predictions = model.predict(X_new)
</code></pre>
        </div>
"""
        
        return html
    
    def _html_footer(self) -> str:
        """Generate HTML footer"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <div class="timestamp">
            Report generated on {timestamp} by AutoML System v1.0
        </div>
    </div>
</body>
</html>
"""
