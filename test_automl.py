"""
Test script for AutoML system - Programmatic usage example
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("AutoML System - Programmatic Test")
print("="*60)

# Import AutoML components
print("\n[1/6] Importing AutoML components...")
try:
    from automl.config.settings import AutoMLConfig
    from automl.data.ingestion import DataIngestor
    from automl.data.profiling import DataProfiler
    from automl.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
    from automl.models.trainer import ModelTrainer
    from automl.reports.report_generator import ReportGenerator
    print("    ✓ All components imported successfully")
except ImportError as e:
    print(f"    ✗ Import error: {e}")
    print("    Run: pip install -r requirements.txt")
    exit(1)

# Create sample data if it doesn't exist
print("\n[2/6] Checking for sample data...")
sample_file = 'sample_data/loan_approval.csv'

if not Path(sample_file).exists():
    print("    Sample data not found. Creating it...")
    import create_sample_data
    print("    ✓ Sample data created")
else:
    print(f"    ✓ Found: {sample_file}")

# Step 1: Ingest data
print("\n[3/6] Ingesting data...")
try:
    ingestor = DataIngestor()
    df, messages = ingestor.ingest(sample_file)
    
    if df is not None:
        print(f"    ✓ Data loaded: {len(df)} rows × {len(df.columns)} columns")
        print(f"    ✓ Messages: {len(messages)}")
    else:
        print("    ✗ Failed to load data")
        for msg in messages:
            print(f"      {msg}")
        exit(1)
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Step 2: Profile data
print("\n[4/6] Profiling data...")
try:
    # Assuming last column is target
    target_col = df.columns[-1]
    print(f"    Target column: '{target_col}'")
    
    profiler = DataProfiler()
    profile = profiler.profile_dataset(df, target_col)
    
    print(f"    ✓ Profile generated")
    print(f"    Task type: {profile.get('target', {}).get('task_type', 'Unknown')}")
    print(f"    Warnings: {len(profile.get('warnings', []))}")
    
    # Show a few warnings
    for warning in profile.get('warnings', [])[:3]:
        print(f"      - {warning}")
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Step 3: Preprocess
print("\n[5/6] Preprocessing data...")
try:
    builder = PreprocessingPipelineBuilder()
    
    # Prepare data
    X, y, warnings = builder.prepare_data(df, target_col)
    print(f"    ✓ Data prepared: {len(X)} rows × {len(X.columns)} features")
    
    # Build pipeline
    pipeline = builder.build_pipeline(X, target_col, profile)
    print(f"    ✓ Pipeline built with {len(builder.preprocessing_steps)} steps")
    
    # Fit and transform
    X_processed = builder.fit_transform(X, y)
    print(f"    ✓ Data transformed")
    
    if hasattr(X_processed, 'shape'):
        print(f"    Final shape: {X_processed.shape}")
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Train models
print("\n[6/6] Training models (fast only)...")
try:
    trainer = ModelTrainer()
    
    # Train with fast models only for quick test
    results = trainer.train_models(X_processed, y, task_type='auto', fast_only=True)
    
    if results.get('best_model_name'):
        print(f"    ✓ Training complete!")
        print(f"    Best model: {results['best_model_name']}")
        print(f"    Best score: {results['best_score']:.4f}")
        print(f"    Models trained: {len(results['all_results'])}")
        
        # Show all models
        print("\n    Model Results:")
        for r in results['all_results']:
            if r.get('status') == 'success':
                print(f"      - {r['model_name']}: {r.get('training_time', 0):.2f}s")
    else:
        print("    ✗ No models trained successfully")
        for warning in results.get('warnings', [])[:5]:
            print(f"      {warning}")
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Generate report
print("\n[BONUS] Generating HTML report...")
try:
    Path('outputs').mkdir(exist_ok=True)
    
    generator = ReportGenerator()
    report_path = generator.generate_report(
        profile=profile,
        preprocessing_steps=builder.preprocessing_steps,
        training_results=results,
        target_col=target_col,
        output_path='outputs/test_report.html'
    )
    
    print(f"    ✓ Report saved: {report_path}")
except Exception as e:
    print(f"    ⚠ Report generation failed: {e}")

# Save model
print("\n[BONUS] Saving model...")
try:
    trainer.save_model('outputs/test_model.pkl')
    print(f"    ✓ Model saved: outputs/test_model.pkl")
except Exception as e:
    print(f"    ⚠ Model save failed: {e}")

# Summary
print("\n" + "="*60)
print("✅ TEST COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  - outputs/test_report.html    (Open in browser)")
print("  - outputs/test_model.pkl      (Saved model)")
print("\nTo use the Streamlit UI:")
print("  streamlit run app.py")
print("\nTo load and use the model:")
print("""
  import joblib
  model_data = joblib.load('outputs/test_model.pkl')
  predictions = model_data['model'].predict(X_new)
""")
print("="*60)
