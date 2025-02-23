# utils/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class DataProcessor:
    def __init__(self):
        # Initialize logging
        self._setup_logging()
        
        # Initialize data storage
        self.data = None
        self.original_data = None  # Keep original data for reset
        
        # Define required columns
        self.required_columns = ['group', 'country', 'continent', 'rating_year', 'start_year']
        
        # Initialize filter tracking
        self.current_filters = {
            'groups': [],
            'countries': [],
            'continents': [],
            'rating_years': [],
            'start_years': []
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def has_data(self):
        """Check if data is loaded"""
        return self.data is not None and not self.data.empty

    def load_file(self, file_path: str):
        """
        Load data from file
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            self.logger.info(f"Loading file: {file_path}")
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load data based on file extension
            if path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Store original data
            self.original_data = self.data.copy()
            
            # Validate and preprocess
            self.validate_and_preprocess_data()
            
            self.logger.info("File loaded successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def validate_and_preprocess_data(self):
        """Validate and preprocess loaded data"""
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        try:
            # Convert data types
            self.data['rating_year'] = pd.to_numeric(self.data['rating_year'], errors='coerce')
            self.data['start_year'] = pd.to_numeric(self.data['start_year'], errors='coerce')
            
            # Ensure string columns are strings and standardize case
            string_columns = ['group', 'country', 'continent']
            for col in string_columns:
                self.data[col] = self.data[col].astype(str).str.strip()
                
            # Remove any rows with NaN values in critical columns
            self.data = self.data.dropna(subset=self.required_columns)
            
            # Store the cleaned data as original
            self.original_data = self.data.copy()
            
            self.logger.info("Data validation and preprocessing completed")
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def apply_filters(self, groups=None, countries=None, continents=None, 
                     rating_years=None, start_years=None):
        """
        Apply filters to the data
        
        Args:
            groups (list): List of selected groups
            countries (list): List of selected countries
            continents (list): List of selected continents
            rating_years (list): List of selected rating years
            start_years (list): List of selected start years
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            if not self.has_data():
                raise ValueError("No data loaded")

            self.logger.info("Applying filters...")
                
            # Store current filters
            self.current_filters = {
                'groups': groups or [],
                'countries': countries or [],
                'continents': continents or [],
                'rating_years': [int(year) for year in (rating_years or [])],
                'start_years': [int(year) for year in (start_years or [])]
            }

            # Start with original data
            filtered_data = self.original_data.copy()

            # Apply each filter if selections were made
            if groups:
                filtered_data = filtered_data[filtered_data['group'].isin(groups)]
                self.logger.info(f"Filtered by groups: {len(filtered_data)} rows remaining")
            
            if countries:
                filtered_data = filtered_data[filtered_data['country'].isin(countries)]
                self.logger.info(f"Filtered by countries: {len(filtered_data)} rows remaining")
            
            if continents:
                filtered_data = filtered_data[filtered_data['continent'].isin(continents)]
                self.logger.info(f"Filtered by continents: {len(filtered_data)} rows remaining")
            
            if rating_years:
                filtered_data = filtered_data[filtered_data['rating_year'].isin(rating_years)]
                self.logger.info(f"Filtered by rating years: {len(filtered_data)} rows remaining")
            
            if start_years:
                filtered_data = filtered_data[filtered_data['start_year'].isin(start_years)]
                self.logger.info(f"Filtered by start years: {len(filtered_data)} rows remaining")

            # Update current data
            self.data = filtered_data
            
            self.logger.info("Filters applied successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error applying filters: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def reset_filters(self):
        """
        Reset all filters and restore original data
        
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            self.logger.info("Resetting filters...")
            
            # Restore original data
            self.data = self.original_data.copy()
            
            # Reset filter tracking
            self.current_filters = {
                'groups': [],
                'countries': [],
                'continents': [],
                'rating_years': [],
                'start_years': []
            }
            
            self.logger.info("Filters reset successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error resetting filters: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def get_filter_summary(self):
        """
        Get summary of currently applied filters
        
        Returns:
            str: Summary of applied filters
        """
        summary = []
        
        for filter_name, values in self.current_filters.items():
            if values:
                summary.append(f"{filter_name.capitalize()}: {', '.join(map(str, values))}")
        
        if not summary:
            return "No filters applied"
        
        return "\n".join(summary)

    def get_unique_values(self, column):
        """
        Get unique values for a given column
        
        Args:
            column (str): Column name
            
        Returns:
            list: Sorted list of unique values
        """
        if self.has_data() and column in self.data.columns:
            return sorted(self.data[column].unique())
        return []

    def get_filtered_data_summary(self):
        """
        Get summary statistics of filtered data
        
        Returns:
            str: Summary statistics text
        """
        if not self.has_data():
            return "No data available"

        summary = []
        summary.append(f"Total Records: {len(self.data):,}")
        
        # Group counts
        if 'group' in self.data.columns:
            group_counts = self.data['group'].value_counts()
            summary.append("\nGroup Distribution:")
            for group, count in group_counts.items():
                summary.append(f"{group}: {count:,}")
        
        # Country counts
        if 'country' in self.data.columns:
            country_counts = self.data['country'].value_counts().head(10)
            summary.append("\nTop 10 Countries:")
            for country, count in country_counts.items():
                summary.append(f"{country}: {count:,}")
        
        # Year distribution
        if 'rating_year' in self.data.columns:
            year_counts = self.data['rating_year'].value_counts().sort_index()
            summary.append("\nRating Year Distribution:")
            for year, count in year_counts.items():
                summary.append(f"{year}: {count:,}")

        return "\n".join(summary)

    def export_filtered_data(self, file_path):
        """
        Export filtered data to file
        
        Args:
            file_path (str): Path where to save the file
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            if not self.has_data():
                raise ValueError("No data available to export")

            self.logger.info(f"Exporting data to: {file_path}")
            
            path = Path(file_path)
            if path.suffix.lower() == '.csv':
                self.data.to_csv(file_path, index=False)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                self.data.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {path.suffix}")
            
            self.logger.info("Data exported successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error exporting data: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def get_data_statistics(self):
        """
        Get comprehensive statistics about the data
        
        Returns:
            dict: Dictionary containing various statistics
        """
        if not self.has_data():
            return {"error": "No data available"}

        try:
            stats = {
                "total_records": len(self.data),
                "total_records_original": len(self.original_data),
                "filtered_percentage": (len(self.data) / len(self.original_data) * 100) if self.original_data is not None else 100,
                "groups": self.data['group'].value_counts().to_dict(),
                "countries": self.data['country'].value_counts().to_dict(),
                "continents": self.data['continent'].value_counts().to_dict(),
                "rating_years": self.data['rating_year'].value_counts().sort_index().to_dict(),
                "start_years": self.data['start_year'].value_counts().sort_index().to_dict(),
                "current_filters": self.current_filters
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            return {"error": str(e)}