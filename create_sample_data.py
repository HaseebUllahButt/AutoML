"""
Create sample test datasets for the AutoML system
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create output directory
Path('sample_data').mkdir(exist_ok=True)

print("Creating test datasets...")

# ============== DATASET 1: Classification (Loan Approval) ==============
print("\n1. Creating loan approval dataset (classification)...")

np.random.seed(42)
n_samples = 1000

loan_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'salary': np.random.randint(30000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'years_employed': np.random.randint(0, 40, n_samples),
    'existing_loans': np.random.randint(0, 5, n_samples),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
    'loan_approved': np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
})

# Add missing values
loan_data.loc[np.random.choice(loan_data.index, 80), 'credit_score'] = np.nan
loan_data.loc[np.random.choice(loan_data.index, 50), 'years_employed'] = np.nan
loan_data.loc[np.random.choice(loan_data.index, 30), 'existing_loans'] = np.nan

# Add edge cases
loan_data.loc[0, 'age'] = '25 years'  # Number with unit
loan_data.loc[1, 'salary'] = '50k'    # Number with k suffix
loan_data.loc[2, 'city'] = ' NYC '    # Whitespace
loan_data.loc[3, 'credit_score'] = '???'  # Missing disguised
loan_data.loc[4, 'education'] = 'bachelor'  # Inconsistent case

# Add some duplicates
loan_data = pd.concat([loan_data, loan_data.iloc[:5]], ignore_index=True)

loan_data.to_csv('sample_data/loan_approval.csv', index=False)
print(f"   ✓ Created: sample_data/loan_approval.csv ({len(loan_data)} rows, {len(loan_data.columns)} columns)")


# ============== DATASET 2: Regression (House Prices) ==============
print("\n2. Creating house prices dataset (regression)...")

np.random.seed(123)
n_houses = 800

house_data = pd.DataFrame({
    'square_feet': np.random.randint(800, 4000, n_houses),
    'bedrooms': np.random.randint(1, 6, n_houses),
    'bathrooms': np.random.randint(1, 5, n_houses),
    'age_years': np.random.randint(0, 100, n_houses),
    'garage_spaces': np.random.randint(0, 4, n_houses),
    'lot_size': np.random.randint(2000, 20000, n_houses),
    'neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural', 'Urban'], n_houses),
    'has_pool': np.random.choice(['Yes', 'No'], n_houses, p=[0.3, 0.7]),
    'school_rating': np.random.randint(1, 11, n_houses),
    'price': np.random.randint(150000, 800000, n_houses)
})

# Add price correlation with features
house_data['price'] = (
    house_data['square_feet'] * 150 +
    house_data['bedrooms'] * 20000 +
    house_data['bathrooms'] * 15000 -
    house_data['age_years'] * 1000 +
    house_data['school_rating'] * 10000 +
    np.random.randint(-50000, 50000, n_houses)
)

# Add missing values
house_data.loc[np.random.choice(house_data.index, 60), 'lot_size'] = np.nan
house_data.loc[np.random.choice(house_data.index, 40), 'age_years'] = np.nan
house_data.loc[np.random.choice(house_data.index, 20), 'school_rating'] = np.nan

# Add edge cases
house_data.loc[0, 'square_feet'] = '2000 sq ft'  # Number with unit
house_data.loc[1, 'has_pool'] = 'yes'  # Different case
house_data.loc[2, 'bathrooms'] = 'N/A'  # Missing disguised
house_data.loc[3, 'neighborhood'] = 'downtown'  # Different case

house_data.to_csv('sample_data/house_prices.csv', index=False)
print(f"   ✓ Created: sample_data/house_prices.csv ({len(house_data)} rows, {len(house_data.columns)} columns)")


# ============== DATASET 3: Small Edge Case Dataset ==============
print("\n3. Creating edge case dataset...")

edge_data = pd.DataFrame({
    'id': range(1, 51),  # ID column (should be removed)
    'mixed_types': [str(i) if i % 3 == 0 else i for i in range(50)],  # Mixed types
    'with_units': ['10kg', '20kg', '5ft', '15kg', '30kg'] * 10,  # Numbers with units
    'missing_various': ['', 'N/A', '???', '--', 'null', None, 'missing', 'Unknown'] * 6 + ['5', '10'],
    'whitespace': [' value ', '  value', 'value  ', 'value', ' value'] * 10,
    'high_cardinality': [f'cat_{i}' for i in range(50)],  # Each value unique
    'constant': ['same'] * 50,  # Constant column
    'target': np.random.choice([0, 1], 50, p=[0.5, 0.5])
})

# Add duplicate rows
edge_data = pd.concat([edge_data, edge_data.iloc[:3]], ignore_index=True)

# Add completely empty row
edge_data.loc[len(edge_data)] = [np.nan] * len(edge_data.columns)

edge_data.to_csv('sample_data/edge_cases.csv', index=False)
print(f"   ✓ Created: sample_data/edge_cases.csv ({len(edge_data)} rows, {len(edge_data.columns)} columns)")


# ============== DATASET 4: Titanic-like (Classic) ==============
print("\n4. Creating titanic-like dataset...")

np.random.seed(456)
n_passengers = 700

titanic_data = pd.DataFrame({
    'PassengerId': range(1, n_passengers + 1),
    'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.2, 0.3, 0.5]),
    'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
    'Age': np.random.randint(1, 80, n_passengers),
    'SibSp': np.random.choice([0, 1, 2, 3], n_passengers, p=[0.5, 0.3, 0.15, 0.05]),
    'Parch': np.random.choice([0, 1, 2], n_passengers, p=[0.7, 0.2, 0.1]),
    'Fare': np.random.uniform(5, 500, n_passengers),
    'Embarked': np.random.choice(['C', 'Q', 'S'], n_passengers, p=[0.2, 0.1, 0.7]),
    'Survived': np.random.choice([0, 1], n_passengers, p=[0.6, 0.4])
})

# Add missing values
titanic_data.loc[np.random.choice(titanic_data.index, 100), 'Age'] = np.nan
titanic_data.loc[np.random.choice(titanic_data.index, 10), 'Embarked'] = np.nan
titanic_data.loc[np.random.choice(titanic_data.index, 5), 'Fare'] = np.nan

titanic_data.to_csv('sample_data/titanic.csv', index=False)
print(f"   ✓ Created: sample_data/titanic.csv ({len(titanic_data)} rows, {len(titanic_data.columns)} columns)")


print("\n" + "="*60)
print("✅ All test datasets created successfully!")
print("="*60)
print("\nAvailable datasets:")
print("  1. sample_data/loan_approval.csv    - Classification (loan approval)")
print("  2. sample_data/house_prices.csv     - Regression (house prices)")
print("  3. sample_data/edge_cases.csv       - Edge cases testing")
print("  4. sample_data/titanic.csv          - Classic Titanic dataset")
print("\nTo test the AutoML system:")
print("  1. Run: streamlit run app.py")
print("  2. Upload any of these CSV files")
print("  3. Follow the UI tabs")
print("\nOr test programmatically with test_automl.py")
