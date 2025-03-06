# utils/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from scipy.stats import chi2_contingency
import sqlite3

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

        # 데이터베이스 폴더 경로 설정
        self.db_folder = Path('./database')
        self.standard_db_path = self.db_folder / 'standard' / 'standard.db'
        self.claim_db_path = self.db_folder / 'claim' / 'claim.db'
        
        # 데이터베이스 폴더가 없으면 생성
        for path in [self.db_folder, self.db_folder / 'standard', self.db_folder / 'claim']:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        
        # SQLite 데이터베이스 연결 초기화
        self._initialize_database_connections()

    def _initialize_database_connections(self):
        """초기 데이터베이스 연결 및 테이블 스키마 확인"""
        # 스탠다드 데이터베이스 초기화
        self._ensure_db_exists(self.standard_db_path)
        
        # 클레임 데이터베이스 초기화
        self._ensure_db_exists(self.claim_db_path)

    def _ensure_db_exists(self, db_path):
        """데이터베이스 파일이 존재하는지 확인하고 없으면 생성"""
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 연결 생성 (파일이 없으면 자동 생성됨)
        conn = sqlite3.connect(db_path)
        conn.close()

    def save_to_database(self, data, data_type, table_name):
        """
        데이터프레임을 SQLite 데이터베이스에 저장
        
        Args:
            data (pandas.DataFrame): 저장할 데이터프레임
            data_type (str): 데이터 타입 ('standard' 또는 'claim')
            table_name (str): 테이블 이름 (사용자 입력)
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            # 테이블 이름 유효성 검사 (SQLite에서 허용되는 문자만 사용)
            import re
            safe_table_name = re.sub(r'[^\w]', '_', table_name)
            
            # 데이터베이스 경로 결정
            db_path = self.standard_db_path if data_type == 'standard' else self.claim_db_path
            
            # 데이터베이스 연결
            conn = sqlite3.connect(db_path)
            
            # 데이터프레임을 SQL 테이블로 저장
            data.to_sql(safe_table_name, conn, if_exists='replace', index=False)
            
            # 메타데이터 테이블이 없으면 생성
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    table_name TEXT PRIMARY KEY,
                    display_name TEXT,
                    created_date TEXT,
                    rows INTEGER,
                    columns INTEGER,
                    column_names TEXT
                )
            ''')
            
            # 메타데이터 삽입 또는 업데이트
            metadata = {
                'table_name': safe_table_name,
                'display_name': table_name,
                'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': ','.join(data.columns)
            }
            
            placeholders = ', '.join(['?'] * len(metadata))
            columns = ', '.join(metadata.keys())
            values = tuple(metadata.values())
            
            conn.execute(f'''
                INSERT OR REPLACE INTO metadata ({columns})
                VALUES ({placeholders})
            ''', values)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"데이터를 '{safe_table_name}' 테이블에 저장했습니다")
            return True, None
            
        except Exception as e:
            error_msg = f"데이터베이스 저장 오류: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
        
    
    def load_from_database(self, data_type, table_name):
        """
        SQLite 데이터베이스에서 데이터 로드
        
        Args:
            data_type (str): 데이터 타입 ('standard' 또는 'claim')
            table_name (str): 테이블 이름
            
        Returns:
            tuple: (success: bool, data: Optional[pandas.DataFrame], error_message: Optional[str])
        """
        try:
            # 데이터베이스 경로 결정
            db_path = self.standard_db_path if data_type == 'standard' else self.claim_db_path
            
            # 데이터베이스 연결
            conn = sqlite3.connect(db_path)
            
            # 테이블 존재 확인
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                return False, None, f"테이블 '{table_name}'이 존재하지 않습니다"
            
            # 데이터 로드
            data = pd.read_sql(f"SELECT * FROM '{table_name}'", conn)
            conn.close()
            
            # 데이터 처리
            if data_type == 'standard':
                self.original_standard_data = data.copy()
                self.standard_data = data.copy()
                self.standard_data = self.process_standard_data(self.standard_data)
                self.active_data_type = 'standard'
            else:
                self.original_claim_data = data.copy()
                self.claim_data = data.copy()
                self.claim_data = self.process_claim_data(self.claim_data)
                self.active_data_type = 'claim'
            
            self.logger.info(f"'{table_name}'에서 {len(data)} 행을 로드했습니다")
            return True, data, None
            
        except Exception as e:
            error_msg = f"데이터베이스 로드 오류: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def get_table_list(self, data_type):
        """
        데이터베이스의 테이블 목록 조회
        
        Args:
            data_type (str): 데이터 타입 ('standard' 또는 'claim')
            
        Returns:
            list: 테이블 정보 목록 [{name, display_name, date, rows}]
        """
        try:
            # 데이터베이스 경로 결정
            db_path = self.standard_db_path if data_type == 'standard' else self.claim_db_path
            
            # 데이터베이스 파일이 없으면 빈 목록 반환
            if not db_path.exists():
                return []
            
            # 데이터베이스 연결
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 메타데이터 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
            if cursor.fetchone():
                # 메타데이터에서 테이블 정보 가져오기
                cursor.execute("""
                    SELECT table_name, display_name, created_date, rows
                    FROM metadata
                    ORDER BY created_date DESC
                """)
                tables = [
                    {
                        'name': row[0],  # 실제 테이블 이름
                        'display_name': row[1],  # 표시용 이름
                        'date': row[2],  # 생성 날짜
                        'rows': row[3]   # 행 수
                    }
                    for row in cursor.fetchall()
                ]
            else:
                # 메타데이터가 없으면 기본 테이블 목록만 가져오기
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'metadata'")
                tables = [
                    {
                        'name': row[0],
                        'display_name': row[0],
                        'date': 'Unknown',
                        'rows': 0
                    }
                    for row in cursor.fetchall()
                ]
            
            conn.close()
            return tables
            
        except Exception as e:
            self.logger.error(f"테이블 목록 조회 오류: {str(e)}")
            return []

    def delete_from_database(self, data_type, table_name):
        """
        Delete a table from the SQLite database
        
        Args:
            data_type (str): Data type ('standard' or 'claim')
            table_name (str): Table name to delete
            
        Returns:
            tuple: (success: bool, error_message: Optional[str])
        """
        try:
            # Determine database path
            db_path = self.standard_db_path if data_type == 'standard' else self.claim_db_path
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                return False, f"Table '{table_name}' does not exist"
            
            # Delete the table
            cursor.execute(f"DROP TABLE IF EXISTS '{table_name}'")
            
            # Remove from metadata if it exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
            if cursor.fetchone():
                cursor.execute("DELETE FROM metadata WHERE table_name=?", (table_name,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Deleted table '{table_name}' from database")
            return True, None
            
        except Exception as e:
            error_msg = f"Error deleting from database: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

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
        """
        Check if data is loaded
        
        Args:
            data_type (str): Type of data ('standard' or 'claim'), if None uses active_data_type
            
        Returns:
            bool: True if data is loaded and not empty, False otherwise
        """
        # 디버깅 정보 추가
        print(f"Checking for {data_type if data_type else 'active'} data")
        
        if data_type is None:
            data_type = self.active_data_type
            print(f"Using active_data_type: {data_type}")
        
        # 명시적으로 데이터 속성에 접근하여 검사
        if data_type == 'standard':
            has_data = self.standard_data is not None and not self.standard_data.empty
            print(f"Standard data check: {has_data}")
            if not has_data:
                print(f"  standard_data is None: {self.standard_data is None}")
                if self.standard_data is not None:
                    print(f"  standard_data is empty: {self.standard_data.empty}")
                    print(f"  standard_data shape: {self.standard_data.shape}")
            return has_data
        
        elif data_type == 'claim':
            has_data = self.claim_data is not None and not self.claim_data.empty
            print(f"Claim data check: {has_data}")
            if not has_data:
                print(f"  claim_data is None: {self.claim_data is None}")
                if self.claim_data is not None:
                    print(f"  claim_data is empty: {self.claim_data.empty}")
                    print(f"  claim_data shape: {self.claim_data.shape}")
            return has_data
        
        print(f"Unknown data type: {data_type}")
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
            
            # Apply filters to standard data
            temp_filtered = filtered_standard.copy()
            
            filter_applied = False
            
            # Apply each filter if selected
            for filter_name, filter_values in [
                ('group', groups),
                ('country', countries),
                ('continent', continents),
                ('rating_year', rating_years),
                ('start_year', start_years)
            ]:
                if filter_values and filter_name in temp_filtered.columns:
                    temp_filtered = temp_filtered[temp_filtered[filter_name].isin(filter_values)]
                    
                    if len(temp_filtered) > 0:
                        filtered_standard = temp_filtered.copy()
                        filter_applied = True
                    else:
                        self.logger.warning(f"{filter_name} filter would remove all data - skipping this filter")
            
            # Final check to ensure we haven't filtered out all data
            if len(filtered_standard) == 0:
                self.logger.warning("All filters combined would remove all data - using original dataset")
                filtered_standard = self.original_standard_data.copy()
                filter_applied = False

            # Update standard data and apply processing
            self.standard_data = self.process_standard_data(filtered_standard)
            
            # Handle claim data filtering
            if claim_data_loaded and filter_applied:
                filtered_claim = self.original_claim_data.copy()
                
                # Add filtering logic for claim data based on standard data filtering
                if 'policy_id' in filtered_standard.columns and 'policy_id' in filtered_claim.columns:
                    filtered_policy_ids = filtered_standard['policy_id'].unique()
                    filtered_claim = filtered_claim[filtered_claim['policy_id'].isin(filtered_policy_ids)]
                
                # Additional cross-dataset filtering if possible
                if countries and 'country' in filtered_claim.columns:
                    filtered_claim = filtered_claim[filtered_claim['country'].isin(countries)]
                
                if groups and 'group' in filtered_claim.columns:
                    filtered_claim = filtered_claim[filtered_claim['group'].isin(groups)]
                
                if continents and 'continent' in filtered_claim.columns:
                    filtered_claim = filtered_claim[filtered_claim['continent'].isin(continents)]
                
                # Final check for claim data
                if len(filtered_claim) == 0:
                    self.logger.warning("Claim filters would remove all data - using original claim dataset")
                    filtered_claim = self.original_claim_data.copy()
                
                # Update claim data and apply processing
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
            
        # Add observed=True to suppress the warning
        return data.groupby('age_group', observed=True)['amount'].mean()

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
        
        # Add observed=True to suppress the warning
        return data.groupby(['age_group', 'gender'], observed=True).size().reset_index(name='count')

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