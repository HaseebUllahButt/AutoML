#!/bin/bash
# Quick start script for AutoML System

echo "=========================================="
echo "🤖 AutoML System - Quick Start"
echo "=========================================="

# Step 1: Create sample data
echo ""
echo "Step 1: Creating sample datasets..."
python3 create_sample_data.py

# Step 2: Test the system
echo ""
echo "Step 2: Testing AutoML system..."
python3 test_automl.py

# Step 3: Launch Streamlit
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To launch the Streamlit UI, run:"
echo "  streamlit run app.py"
echo ""
echo "Then upload one of these test files:"
echo "  - sample_data/loan_approval.csv"
echo "  - sample_data/house_prices.csv"
echo "  - sample_data/titanic.csv"
echo ""
echo "Or check the generated files:"
echo "  - outputs/test_report.html (open in browser)"
echo "  - outputs/test_model.pkl (saved model)"
echo ""
