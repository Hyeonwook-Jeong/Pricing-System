import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define sample size
n_samples = 10000

# Create base data
data = {
    'date': pd.date_range(start='2015-01-01', periods=n_samples, freq='D'),
    'age': np.random.randint(0, 100, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'diagnosis': np.random.choice([
        'Alzheimer\'s', 'Type 2 Diabetes', 'Cataracts', 'COPD',
        'Arthritis', 'Asthma', 'Hypertension', 'Heart Disease',
        'Depression', 'Osteoporosis'
    ], n_samples),
    'amount': np.random.uniform(100, 15000, n_samples).round(2),
    
    # New columns based on your filters
    'group': np.random.choice([
        'Singapore Individual', 'Singapore Corporate',
        'Dubai Individual', 'Summit', 'EHP'
    ], n_samples),
    
    'country': np.random.choice([
        'United States', 'China', 'Japan', 'Germany',
        'India', 'United Kingdom', 'France', 'Italy',
        'Canada', 'Brazil', 'Russia', 'South Korea',
        'Australia', 'Spain', 'Mexico', 'Indonesia',
        'Netherlands', 'Saudi Arabia', 'Switzerland', 'Turkey'
    ], n_samples),
    
    'continent': np.random.choice([
        'Asia', 'Europe', 'North America', 'South America',
        'Africa', 'Oceania', 'Antarctica'
    ], n_samples),
    
    'rating_year': np.random.choice(range(2015, 2025), n_samples),
    'start_year': np.random.choice(range(2015, 2025), n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure start_year is not greater than rating_year
mask = df['start_year'] > df['rating_year']
df.loc[mask, 'start_year'] = df.loc[mask, 'rating_year']

# Format the date column
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Adjust amount to have some variety in decimal places
df['amount'] = df['amount'].apply(lambda x: round(x, np.random.choice([2, 4, 8])))

# Save to CSV with UTF-8 encoding
df.to_csv('insurance_claims_10k.csv', index=False, encoding='utf-8')

# Display first few rows and basic statistics
print("First few rows of the generated data:")
print(df.head().to_string())
print("\nDataset Shape:", df.shape)
print("\nNumber of unique values per column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")