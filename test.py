import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for past 2 years
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dates = []
current_date = start_date
while current_date <= end_date:
    # Add more claims during winter months
    num_claims = np.random.poisson(
        lam=10 if current_date.month in [12, 1, 2] else 5
    )
    dates.extend([current_date] * num_claims)
    current_date += timedelta(days=1)

# Common medical conditions with rough age associations
age_related_conditions = {
    'young': [
        'Common Cold', 'Flu', 'Sports Injury', 'Allergies', 
        'Asthma', 'Appendicitis'
    ],
    'adult': [
        'Hypertension', 'Type 2 Diabetes', 'Back Pain', 'Anxiety',
        'Depression', 'Migraine'
    ],
    'elderly': [
        'Arthritis', 'Heart Disease', 'COPD', 'Osteoporosis',
        'Alzheimer\'s', 'Cataracts'
    ]
}

def get_diagnosis(age):
    """Return appropriate diagnosis based on age"""
    if age < 30:
        return random.choice(age_related_conditions['young'])
    elif age < 60:
        return random.choice(age_related_conditions['adult'])
    else:
        return random.choice(age_related_conditions['elderly'])

def get_amount(diagnosis):
    """Generate reasonable claim amount based on condition"""
    base_amounts = {
        'Common Cold': (50, 200),
        'Flu': (100, 500),
        'Sports Injury': (500, 5000),
        'Allergies': (50, 300),
        'Asthma': (200, 1000),
        'Appendicitis': (5000, 20000),
        'Hypertension': (200, 1000),
        'Type 2 Diabetes': (300, 2000),
        'Back Pain': (500, 3000),
        'Anxiety': (200, 1000),
        'Depression': (200, 1500),
        'Migraine': (100, 800),
        'Arthritis': (300, 2000),
        'Heart Disease': (5000, 50000),
        'COPD': (1000, 5000),
        'Osteoporosis': (500, 3000),
        'Alzheimer\'s': (2000, 10000),
        'Cataracts': (3000, 8000)
    }
    
    min_amount, max_amount = base_amounts.get(diagnosis, (100, 1000))
    return round(random.uniform(min_amount, max_amount), 2)

# Generate the dataset
num_records = len(dates)
data = {
    'date': dates,
    'age': np.random.normal(45, 20, num_records).clip(0, 100).astype(int),
    'gender': np.random.choice(['M', 'F'], num_records),
}

# Add diagnosis based on age
data['diagnosis'] = [get_diagnosis(age) for age in data['age']]

# Add claim amounts based on diagnosis
data['amount'] = [get_amount(diag) for diag in data['diagnosis']]

# Create DataFrame
df = pd.DataFrame(data)

# Add some seasonal variation to claim amounts
df['month'] = df['date'].dt.month
winter_mask = df['month'].isin([12, 1, 2])
df.loc[winter_mask, 'amount'] *= 1.2  # 20% higher claims in winter

# Add some age-based variation
df.loc[df['age'] > 60, 'amount'] *= 1.3  # 30% higher claims for elderly

# Add some gender-based variation (for example purposes)
df.loc[df['gender'] == 'F', 'amount'] *= 1.1  # 10% higher claims for females

# Sort by date
df = df.sort_values('date').drop('month', axis=1)

# Save to CSV
df.to_csv('test_insurance_claims.csv', index=False)

# Print sample and summary
print("\nFirst few records:")
print(df.head())

print("\nDataset Summary:")
print(f"Total number of records: {len(df)}")
print("\nSummary statistics:")
print(df.describe())

print("\nDiagnosis distribution:")
print(df['diagnosis'].value_counts().head())

print("\nMonthly claim counts:")
print(df.groupby(df['date'].dt.to_period('M')).size().head())

print("\nAverage claim amount by age group:")
age_bins = [0, 18, 30, 45, 60, 75, 100]
age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
print(df.groupby('age_group')['amount'].mean())