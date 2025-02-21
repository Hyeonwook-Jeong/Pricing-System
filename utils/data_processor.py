# utils/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path

class DataProcessor:
    def __init__(self):
        self.data = None
        self.required_columns = ['age', 'amount', 'diagnosis', 'date', 'gender']

    def has_data(self):
        """Check if data is loaded"""
        return self.data is not None

    def load_file(self, file_path: str):
        """Load data from file"""
        try:
            path = Path(file_path)
            
            if path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)

            self.validate_and_preprocess_data()
            return True, None
        except Exception as e:
            return False, str(e)

    def validate_and_preprocess_data(self):
        """Validate and preprocess loaded data"""
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert data types
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')
        self.data['amount'] = pd.to_numeric(self.data['amount'], errors='coerce')
        
        # Ensure gender is uppercase
        self.data['gender'] = self.data['gender'].str.upper()

        # Remove any rows with NaN values in critical columns
        self.data = self.data.dropna(subset=['age', 'amount', 'date', 'gender'])

        # Add derived columns
        self.add_derived_columns()

    def add_derived_columns(self):
        """Add derived columns for analysis"""
        # Age groups
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
        self.data['age_group'] = pd.cut(self.data['age'], 
                                      bins=age_bins, 
                                      labels=age_labels,
                                      ordered=True)

        # Seasonal data
        self.data['month'] = self.data['date'].dt.month
        self.data['season'] = pd.cut(
            self.data['date'].dt.month,
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            ordered=True
        )

    def get_data(self):
        """Get the processed data"""
        return self.data if self.has_data() else None

    def get_data_summary(self):
        """Get summary of the data"""
        if not self.has_data():
            return "No data loaded"

        summary = []
        summary.append(f"Total Records: {len(self.data):,}")
        summary.append(f"Total Columns: {len(self.data.columns)}")
        summary.append(f"Columns: {', '.join(self.data.columns)}")
        
        # Basic statistics
        numeric_stats = self.data.describe().round(2)
        summary.append("\nNumeric Statistics:")
        summary.append(str(numeric_stats))
        
        # Sample data
        summary.append("\nFirst 5 Rows:")
        summary.append(str(self.data.head()))

        return "\n".join(summary)

    def get_missing_values_summary(self):
        """Get summary of missing values"""
        if not self.has_data():
            return {}
        return dict(self.data[self.required_columns].isna().sum())

    def get_age_distribution(self):
        """Get age distribution data"""
        if not self.has_data():
            return None
        return self.data['age_group'].value_counts(sort=False)

    def get_amount_distribution(self):
        """Get amount distribution data"""
        if not self.has_data():
            return None
        return self.data['amount'].dropna()

    def get_diagnosis_distribution(self):
        """Get diagnosis distribution data"""
        if not self.has_data():
            return None
        return self.data['diagnosis'].value_counts().head(10)

    def get_monthly_trend(self):
        """Get monthly trend data"""
        if not self.has_data():
            return None
        monthly_data = self.data.groupby(self.data['date'].dt.to_period('M')).size()
        return monthly_data.sort_index()

    def get_yearly_trend(self):
        """Get yearly trend data"""
        if not self.has_data():
            return None
        yearly_data = self.data.groupby(self.data['date'].dt.year).agg({
            'amount': ['count', 'mean']
        })
        return yearly_data.sort_index()

    def get_seasonal_pattern(self):
        """Get seasonal pattern data"""
        if not self.has_data():
            return None
        return self.data.groupby('season', observed=True).agg({
            'amount': ['count', 'mean']
        })

    def get_correlation_matrix(self):
        """Get correlation matrix for numeric columns"""
        if not self.has_data():
            return None
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        return self.data[numeric_cols].corr()

    def get_average_amount_by_age(self):
        """Get average amount by age group"""
        if not self.has_data():
            return None
        return self.data.groupby('age_group', observed=True)['amount'].mean().sort_index()

    def get_gender_distribution(self):
        """Get gender distribution data by age group"""
        if not self.has_data():
            return None
        
        # Calculate counts by age_group and gender
        gender_dist = self.data.groupby(['age_group', 'gender'], observed=True).size().unstack(fill_value=0)
        
        # Reset index to make age_group a column
        gender_dist = gender_dist.reset_index()
        
        # Ensure M and F columns exist
        if 'M' not in gender_dist.columns:
            gender_dist['M'] = 0
        if 'F' not in gender_dist.columns:
            gender_dist['F'] = 0
            
        # Melt the DataFrame to get it in the right format
        gender_dist = gender_dist.melt(
            id_vars=['age_group'],
            value_vars=['M', 'F'],
            var_name='gender',
            value_name='count'
        )
        
        return gender_dist