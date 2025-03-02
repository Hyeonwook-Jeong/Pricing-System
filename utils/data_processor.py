# utils/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from scipy.stats import chi2_contingency

def cramers_v(confusion_matrix):
    """
    Calculate Cramer's V correlation coefficient for categorical variables
    
    Args:
        confusion_matrix (numpy.ndarray): Contingency matrix
    
    Returns:
        float: Cramer's V coefficient
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

class DataProcessor:
    def __init__(self):
        # Initialize logging
        self._setup_logging()
        
        # Initialize separate data storage for standard and claim
        self.standard_data = None
        self.original_standard_data = None
        self.claim_data = None
        self.original_claim_data = None
        
        # Current active data type
        self.active_data_type = None  # 'standard' or 'claim'
        
        # Define required columns
        self.required_columns = {
            'standard': ['group', 'country', 'continent', 'rating_year', 'start_year'],
            'claim': ['date', 'amount', 'diagnosis', 'age', 'gender']
        }
        
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

    def has_data(self, data_type=None):
        """Check if data is loaded"""
        if data_type is None:
            data_type = self.active_data_type
        
        if data_type == 'standard':
            return self.standard_data is not None and not self.standard_data.empty
        elif data_type == 'claim':
            return self.claim_data is not None and not self.claim_data.empty
        return False

    def get_data(self, data_type=None):
        """Return current data"""
        if data_type is None:
            data_type = self.active_data_type
            
        if data_type == 'standard':
            return self.standard_data
        elif data_type == 'claim':
            return self.claim_data
        return None

    def load_file(self, file_path: str, data_type: str):
        """
        Load data from file
        
        Args:
            file_path (str): Path to the data file
            data_type (str): Type of data ('standard' or 'claim')
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            self.logger.info(f"Loading {data_type} file: {file_path}")
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load data based on file extension
            if path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Validate data
            self.validate_data(data, data_type)
            
            # Store data and process based on type
            if data_type == 'standard':
                self.original_standard_data = data.copy()
                self.standard_data = data.copy()
                # Apply standard-specific processing
                self.standard_data = self.process_standard_data(self.standard_data)
                self.active_data_type = 'standard'
            elif data_type == 'claim':
                self.original_claim_data = data.copy()
                self.claim_data = data.copy()
                # Apply claim-specific processing
                self.claim_data = self.process_claim_data(self.claim_data)
                self.active_data_type = 'claim'
            
            self.logger.info(f"{data_type} file loaded successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error loading {data_type} file: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def validate_data(self, data, data_type):
        """
        Validate loaded data
        
        Args:
            data (pandas.DataFrame): Data to validate
            data_type (str): Type of data ('standard' or 'claim')
        """
        # Check required columns for the specific data type
        required_cols = self.required_columns.get(data_type, [])
        missing_columns = [col for col in required_cols if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {data_type} data: {', '.join(missing_columns)}")

    def process_standard_data(self, data=None):
        """
        Process standard data with specific transformations
        
        Args:
            data (pandas.DataFrame, optional): Data to process. If None, uses self.standard_data
            
        Returns:
            pandas.DataFrame: Processed standard data
        """
        if data is None:
            if self.standard_data is None:
                return None
            data = self.standard_data.copy()
        
        try:
            # Apply standard-specific processing
            if 'rating_year' in data.columns:
                data['rating_year'] = pd.to_numeric(data['rating_year'], errors='coerce')
            
            if 'start_year' in data.columns:
                data['start_year'] = pd.to_numeric(data['start_year'], errors='coerce')
            
            # Ensure string columns are strings and standardize case
            string_columns = ['group', 'country', 'continent']
            for col in string_columns:
                if col in data.columns:
                    data[col] = data[col].astype(str).str.strip()
                    
            # Remove any rows with NaN values in critical columns
            required_cols = self.required_columns.get('standard', [])
            existing_cols = [col for col in required_cols if col in data.columns]
            if existing_cols:
                data.dropna(subset=existing_cols, inplace=True)
            
            # Add any additional standard-specific processing here
            
            return data
        except Exception as e:
            self.logger.error(f"Error processing standard data: {str(e)}")
            return None

    def process_claim_data(self, data=None):
        """
        Process claim data with specific transformations
        
        Args:
            data (pandas.DataFrame, optional): Data to process. If None, uses self.claim_data
            
        Returns:
            pandas.DataFrame: Processed claim data
        """
        if data is None:
            if self.claim_data is None:
                return None
            data = data.copy()
        
        try:
            # Apply claim-specific processing
            if 'date' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                    
            if 'amount' in data.columns:
                data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
            
            # Create age groups if needed
            if 'age_group' not in data.columns and 'age' in data.columns:
                age_bins = [0, 18, 30, 45, 60, 100]
                age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
                data['age_group'] = pd.cut(
                    data['age'], 
                    bins=age_bins, 
                    labels=age_labels, 
                    right=False
                )
            
            # Create season column if not exists
            if 'season' not in data.columns and 'date' in data.columns:
                try:
                    data['season'] = data['date'].dt.month.map({
                        12: 'Winter', 1: 'Winter', 2: 'Winter',
                        3: 'Spring', 4: 'Spring', 5: 'Spring',
                        6: 'Summer', 7: 'Summer', 8: 'Summer',
                        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
                    })
                except:
                    pass
            
            # Remove any rows with NaN values in critical columns
            required_cols = self.required_columns.get('claim', [])
            existing_cols = [col for col in required_cols if col in data.columns]
            if existing_cols:
                data.dropna(subset=existing_cols, inplace=True)
            
            return data
        except Exception as e:
            self.logger.error(f"Error processing claim data: {str(e)}")
            return None

    def apply_filters_to_both(self, groups=None, countries=None, continents=None, 
                         rating_years=None, start_years=None):
        """
        Apply filters to both standard and claim data
        
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
            # Check if standard data is loaded
            if not self.has_data('standard'):
                raise ValueError("No standard data loaded")
            
            # Check if claim data is loaded
            claim_data_loaded = self.has_data('claim')
            
            self.logger.info("Applying filters to both standard and claim data...")
                
            # Save current filters
            self.current_filters = {
                'groups': groups or [],
                'countries': countries or [],
                'continents': continents or [],
                'rating_years': [int(year) for year in (rating_years or []) if year],
                'start_years': [int(year) for year in (start_years or []) if year]
            }

            # Start with original standard data
            filtered_standard = self.original_standard_data.copy()

            # Apply each filter if selected
            if groups:
                filtered_standard = filtered_standard[filtered_standard['group'].isin(groups)]
                self.logger.info(f"Filtered by groups: {len(filtered_standard)} rows remaining")
            
            if countries:
                filtered_standard = filtered_standard[filtered_standard['country'].isin(countries)]
                self.logger.info(f"Filtered by countries: {len(filtered_standard)} rows remaining")
            
            if continents:
                filtered_standard = filtered_standard[filtered_standard['continent'].isin(continents)]
                self.logger.info(f"Filtered by continents: {len(filtered_standard)} rows remaining")
            
            if rating_years:
                filtered_standard = filtered_standard[filtered_standard['rating_year'].isin(rating_years)]
                self.logger.info(f"Filtered by rating years: {len(filtered_standard)} rows remaining")
            
            if start_years:
                filtered_standard = filtered_standard[filtered_standard['start_year'].isin(start_years)]
                self.logger.info(f"Filtered by start years: {len(filtered_standard)} rows remaining")

            # Update standard data and apply standard-specific processing
            self.standard_data = self.process_standard_data(filtered_standard)
            
            # Handle claim data if loaded
            if claim_data_loaded:
                # Try to filter claim data based on common keys or columns
                filtered_claim = self.original_claim_data.copy()
                
                # If there's a common key between datasets, use it to filter claims
                if 'policy_id' in filtered_standard.columns and 'policy_id' in filtered_claim.columns:
                    filtered_policy_ids = filtered_standard['policy_id'].unique()
                    filtered_claim = filtered_claim[filtered_claim['policy_id'].isin(filtered_policy_ids)]
                    self.logger.info(f"Applied filters to claim data via policy_id: {len(filtered_claim)} rows")
                # If country is a common column and was used in filtering
                elif countries and 'country' in filtered_claim.columns:
                    filtered_claim = filtered_claim[filtered_claim['country'].isin(countries)]
                    self.logger.info(f"Applied filters to claim data via country: {len(filtered_claim)} rows")
                
                # Update claim data and apply claim-specific processing
                self.claim_data = self.process_claim_data(filtered_claim)
            
            self.logger.info("Filters applied successfully to both datasets")
            return True, None
            
        except Exception as e:
            error_msg = f"Error applying filters: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def reset_filters(self):
        """
        Reset all filters
        
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            self.logger.info("Resetting all filters")
            
            # Reset current filters
            self.current_filters = {
                'groups': [],
                'countries': [],
                'continents': [],
                'rating_years': [],
                'start_years': []
            }
            
            # Restore original data
            if self.original_standard_data is not None:
                self.standard_data = self.original_standard_data.copy()
                # Apply standard processing to restored data
                self.standard_data = self.process_standard_data(self.standard_data)
            
            if self.original_claim_data is not None:
                self.claim_data = self.original_claim_data.copy()
                # Apply claim processing to restored data
                self.claim_data = self.process_claim_data(self.claim_data)
            
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

    def get_unique_values(self, column, data_type=None):
        """
        Get unique values for a given column
        
        Args:
            column (str): Column name
            data_type (str): Type of data ('standard' or 'claim')
            
        Returns:
            list: Sorted list of unique values
        """
        try:
            self.logger.info(f"Getting unique values for column: {column}, data_type: {data_type}")
            
            if data_type is None:
                data_type = self.active_data_type
                
            data = self.get_data(data_type)
            
            if data is not None and column in data.columns:
                unique_values = sorted(data[column].unique())
                self.logger.info(f"Found {len(unique_values)} unique values for {column}")
                return unique_values
            else:
                self.logger.warning(f"Column {column} not found in {data_type} data")
                return []
        except Exception as e:
            self.logger.error(f"Error getting unique values: {str(e)}")
            return []

    def get_filtered_data_summary(self, data_type=None):
        """
        Get summary statistics of filtered data
        
        Args:
            data_type (str): Type of data ('standard' or 'claim')
            
        Returns:
            str: Summary statistics text
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return f"No {data_type} data available"

        summary = []
        summary.append(f"Total Records: {len(data):,}")
        
        if data_type == 'standard':
            # Group counts
            if 'group' in data.columns:
                group_counts = data['group'].value_counts()
                summary.append("\nGroup Distribution:")
                for group, count in group_counts.items():
                    summary.append(f"{group}: {count:,}")
            
            # Country counts
            if 'country' in data.columns:
                country_counts = data['country'].value_counts().head(10)
                summary.append("\nTop 10 Countries:")
                for country, count in country_counts.items():
                    summary.append(f"{country}: {count:,}")
            
            # Year distribution
            if 'rating_year' in data.columns:
                year_counts = data['rating_year'].value_counts().sort_index()
                summary.append("\nRating Year Distribution:")
                for year, count in year_counts.items():
                    summary.append(f"{year}: {count:,}")
        
        elif data_type == 'claim':
            # Age group distribution
            if 'age_group' in data.columns:
                age_counts = data['age_group'].value_counts()
                summary.append("\nAge Group Distribution:")
                for age, count in age_counts.items():
                    summary.append(f"{age}: {count:,}")
            
            # Diagnosis distribution
            if 'diagnosis' in data.columns:
                diag_counts = data['diagnosis'].value_counts().head(10)
                summary.append("\nTop 10 Diagnoses:")
                for diag, count in diag_counts.items():
                    summary.append(f"{diag}: {count:,}")
            
            # Amount statistics
            if 'amount' in data.columns:
                summary.append("\nAmount Statistics:")
                summary.append(f"Average: ${data['amount'].mean():,.2f}")
                summary.append(f"Median: ${data['amount'].median():,.2f}")
                summary.append(f"Min: ${data['amount'].min():,.2f}")
                summary.append(f"Max: ${data['amount'].max():,.2f}")

        return "\n".join(summary)

    def export_filtered_data(self, file_path, data_type=None):
        """
        Export filtered data to file
        
        Args:
            file_path (str): Path where to save the file
            data_type (str): Type of data ('standard' or 'claim')
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        try:
            if data is None:
                raise ValueError(f"No {data_type} data available to export")

            self.logger.info(f"Exporting {data_type} data to: {file_path}")
            
            path = Path(file_path)
            if path.suffix.lower() == '.csv':
                data.to_csv(file_path, index=False)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                data.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {path.suffix}")
            
            self.logger.info(f"{data_type} data exported successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error exporting data: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def get_data_statistics(self, data_type=None):
        """
        Get comprehensive statistics about the data
        
        Args:
            data_type (str): Type of data ('standard' or 'claim')
            
        Returns:
            dict: Dictionary containing various statistics
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        original_data = self.original_standard_data if data_type == 'standard' else self.original_claim_data
        
        if data is None:
            return {"error": f"No {data_type} data available"}

        try:
            stats = {
                "data_type": data_type,
                "total_records": len(data),
                "total_records_original": len(original_data) if original_data is not None else 0,
                "filtered_percentage": (len(data) / len(original_data) * 100) if original_data is not None else 100,
            }
            
            if data_type == 'standard':
                stats.update({
                    "groups": data['group'].value_counts().to_dict() if 'group' in data.columns else {},
                    "countries": data['country'].value_counts().to_dict() if 'country' in data.columns else {},
                    "continents": data['continent'].value_counts().to_dict() if 'continent' in data.columns else {},
                    "rating_years": data['rating_year'].value_counts().sort_index().to_dict() if 'rating_year' in data.columns else {},
                    "start_years": data['start_year'].value_counts().sort_index().to_dict() if 'start_year' in data.columns else {},
                    "current_filters": self.current_filters
                })
            
            elif data_type == 'claim':
                stats.update({
                    "age_groups": data['age_group'].value_counts().to_dict() if 'age_group' in data.columns else {},
                    "diagnoses": data['diagnosis'].value_counts().head(10).to_dict() if 'diagnosis' in data.columns else {},
                    "gender": data['gender'].value_counts().to_dict() if 'gender' in data.columns else {},
                    "amount_stats": {
                        "mean": data['amount'].mean() if 'amount' in data.columns else None,
                        "median": data['amount'].median() if 'amount' in data.columns else None,
                        "min": data['amount'].min() if 'amount' in data.columns else None,
                        "max": data['amount'].max() if 'amount' in data.columns else None
                    }
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            return {"error": str(e)}

    def get_correlation_matrix(self, data_type=None):
        """
        Calculate correlation matrix for categorical variables using Cramer's V
        
        Args:
            data_type (str): Type of data ('standard' or 'claim')
            
        Returns:
            pandas.DataFrame: Correlation matrix
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Select appropriate categorical columns based on data type
        if data_type == 'standard':
            cat_columns = ['group', 'country', 'continent']
        else:  # claim
            cat_columns = ['age_group', 'gender', 'diagnosis', 'season']
        
        # Ensure all selected columns exist in the dataframe
        available_columns = [col for col in cat_columns if col in data.columns]
        
        if len(available_columns) < 2:
            return None
        
        # Preprocessing: ensure age_group exists if age is in available columns
        if 'age' in available_columns and 'age_group' not in available_columns:
            age_bins = [0, 18, 30, 45, 60, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
            data['age_group'] = pd.cut(
                data['age'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
            available_columns = [col if col != 'age' else 'age_group' for col in available_columns]
        
        # Create correlation matrix
        correlation_matrix = pd.DataFrame(
            index=available_columns, 
            columns=available_columns, 
            dtype=float
        )
        
        # Calculate Cramer's V for each pair of categorical variables
        for i, col1 in enumerate(available_columns):
            for j, col2 in enumerate(available_columns):
                if i == j:
                    correlation_matrix.loc[col1, col2] = 1.0
                elif i < j:
                    # Create contingency table
                    contingency_table = pd.crosstab(data[col1], data[col2])
                    
                    # Calculate Cramer's V
                    cramer_v = cramers_v(contingency_table.values)
                    
                    # Store in symmetric matrix
                    correlation_matrix.loc[col1, col2] = cramer_v
                    correlation_matrix.loc[col2, col1] = cramer_v
        
        return correlation_matrix

    def get_age_distribution(self, data_type=None):
        """
        Calculate age distribution
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.Series: Age group distribution
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Ensure age groups exist
        if 'age_group' not in data.columns and 'age' in data.columns:
            age_bins = [0, 18, 30, 45, 60, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
            data['age_group'] = pd.cut(
                data['age'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
        
        if 'age_group' in data.columns:
            return data['age_group'].value_counts().sort_index()
        
        return None

    def get_amount_distribution(self, data_type=None):
        """
        Get distribution of amounts
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            numpy.array or None: Array of amounts
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Return amount column if it exists
        return data.get('amount', pd.Series()).values

    def get_diagnosis_distribution(self, data_type=None):
        """
        Calculate diagnosis distribution
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.Series: Diagnosis distribution
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None or 'diagnosis' not in data.columns:
            return None
        
        return data['diagnosis'].value_counts().head(10)

    def get_monthly_trend(self, data_type=None):
        """
        Calculate monthly trend
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.Series: Monthly trend
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Extract monthly trend from date
        if 'date' in data.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    dates = pd.to_datetime(data['date'], errors='coerce')
                else:
                    dates = data['date']
                return dates.dt.to_period('M').value_counts().sort_index()
            except Exception as e:
                self.logger.error(f"Error processing date column: {e}")
                return None
        elif 'month' in data.columns:
            return data['month'].value_counts().sort_index()
        else:
            return None

    def get_yearly_trend(self, data_type=None):
        """
        Calculate yearly trend with amount statistics
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.DataFrame: Yearly trend with count and mean amount
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Determine year column based on data type
        if data_type == 'standard':
            year_col = 'rating_year' if 'rating_year' in data.columns else 'year'
        else:  # claim
            year_col = 'year'
            if year_col not in data.columns and 'date' in data.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(data['date']):
                        data['year'] = pd.to_datetime(data['date'], errors='coerce').dt.year
                    else:
                        data['year'] = data['date'].dt.year
                    year_col = 'year'
                except:
                    return None
        
        # Check if year column exists
        if year_col not in data.columns:
            return None
            
        # Get amount column if it exists
        amount_col = 'amount' if 'amount' in data.columns else None
        
        if amount_col:
            return data.groupby(year_col).agg({
                amount_col: ['count', 'mean']
            })
        else:
            return pd.DataFrame(data[year_col].value_counts().sort_index())

    def get_average_amount_by_age(self, data_type=None):
        """
        Calculate average amount by age group
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.Series: Average amount by age group
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Create age groups if needed
        if 'age_group' not in data.columns and 'age' in data.columns:
            age_bins = [0, 18, 30, 45, 60, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
            data['age_group'] = pd.cut(
                data['age'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
        
        # Check for age_group and amount columns
        if 'age_group' not in data.columns or 'amount' not in data.columns:
            return None
            
        return data.groupby('age_group')['amount'].mean()

    def get_gender_distribution(self, data_type=None):
        """
        Calculate gender distribution by age group
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.DataFrame: Gender distribution
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Create age groups if needed
        if 'age_group' not in data.columns and 'age' in data.columns:
            age_bins = [0, 18, 30, 45, 60, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
            data['age_group'] = pd.cut(
                data['age'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
        
        # Check for required columns
        if 'gender' not in data.columns or 'age_group' not in data.columns:
            return None
        
        return data.groupby(['age_group', 'gender']).size().reset_index(name='count')

    def get_seasonal_pattern(self, data_type=None):
        """
        Calculate seasonal pattern
        
        Args:
            data_type (str): Type of data (default to active type)
            
        Returns:
            pandas.DataFrame: Seasonal pattern with count and mean amount
        """
        if data_type is None:
            data_type = self.active_data_type
            
        data = self.get_data(data_type)
        
        if data is None:
            return None
        
        # Add season column if not exists
        if 'season' not in data.columns and 'date' in data.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    date_col = pd.to_datetime(data['date'], errors='coerce')
                else:
                    date_col = data['date']
                    
                data['season'] = date_col.dt.month.map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
                })
            except Exception as e:
                self.logger.error(f"Error processing date column: {e}")
                return None
        
        # Check for required columns
        if 'season' not in data.columns:
            return None
            
        # Get amount column if it exists
        amount_col = 'amount' if 'amount' in data.columns else None
        
        if amount_col:
            return data.groupby('season').agg({
                amount_col: ['count', 'mean']
            })
        else:
            return pd.DataFrame(data['season'].value_counts())