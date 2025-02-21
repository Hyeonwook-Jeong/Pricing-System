import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import seaborn as sns
import traceback
import logging
from pathlib import Path
from PIL import Image, ImageTk
from functools import wraps
from typing import Dict, List, Tuple, Any, Optional

class Config:
    # Window settings
    WINDOW_TITLE = "GPS"
    WINDOW_SIZE = "1200x800"
    
    # Theme settings
    APPEARANCE_MODE = "dark"
    COLOR_THEME = "blue"
    
    # Graph settings
    GRAPH_SETTINGS = {
        'figsize': (8, 6),
        'dpi': 100,
        'facecolor': '#2B2B2B'
    }
    
    # Style settings
    STYLES = {
        'bg_color': '#2B2B2B',
        'text_color': 'white',
        'accent_color': 'lightblue'
    }
    
    # Data settings
    REQUIRED_COLUMNS = ['age', 'amount', 'diagnosis', 'date']
    
    # Visualization settings
    AGE_BINS = [0, 18, 30, 45, 60, 75, 100]
    AGE_LABELS = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
    SEASONS = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }

class DataProcessor:
    def __init__(self):
        self.data = None
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        try:
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)
            
            self._validate_columns()
            self._preprocess_data()
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_columns(self):
        """Validate required columns exist in the dataset."""
        missing_columns = [col for col in Config.REQUIRED_COLUMNS 
                         if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def _preprocess_data(self):
        """Preprocess the loaded data."""
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')
        self.data['amount'] = pd.to_numeric(self.data['amount'], errors='coerce')
        
        # Add derived columns
        self.data['age_group'] = pd.cut(self.data['age'], 
                                      bins=Config.AGE_BINS, 
                                      labels=Config.AGE_LABELS)
        self.data['month'] = self.data['date'].dt.month
        self.data['season'] = pd.cut(self.data['month'],
                                   bins=[0, 3, 6, 9, 12],
                                   labels=['Winter', 'Spring', 'Summer', 'Fall'])

    def get_data_summary(self) -> str:
        """Generate data summary text."""
        if self.data is None:
            return "No data loaded"
            
        info_text = f"Dataset Information:\n{'='*50}\n"
        info_text += f"Total Records: {len(self.data):,}\n"
        info_text += f"Total Columns: {len(self.data.columns)}\n"
        info_text += f"Columns: {', '.join(self.data.columns)}\n\n"
        
        info_text += f"Basic Statistics:\n{'='*50}\n"
        stats = self.data.describe().round(2)
        info_text += stats.to_string()
        info_text += f"\n\n"
        
        info_text += f"First 5 Rows of Data:\n{'='*50}\n"
        info_text += self.data.head().to_string()
        
        return info_text

def safe_plot(func):
    """Decorator for safe plot execution with error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Plot error in {func.__name__}: {str(e)}")
            ax = kwargs.get('ax')
            if ax:
                ax.clear()
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                       ha='center', va='center', color='red')
    return wrapper

class VisualizationManager:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.logger = logging.getLogger(__name__)

    @safe_plot
    def plot_age_distribution(self, ax, data_type="claims"):
        """Plot age distribution."""
        if self.data_processor.data is None:
            return
            
        age_dist = self.data_processor.data['age_group'].value_counts()
        ax.bar(age_dist.index, age_dist.values, color='skyblue')
        ax.set_xlabel('Age Group', color='white')
        ax.set_ylabel(f'Number of {data_type.capitalize()}', color='white')
        plt.setp(ax.get_xticklabels(), rotation=45)

    @safe_plot
    def plot_amount_distribution(self, ax, data_type="claims"):
        """Plot amount distribution."""
        if self.data_processor.data is None:
            return
            
        ax.hist(self.data_processor.data['amount'], bins=50, color='lightgreen')
        ax.set_xlabel(f'{data_type.capitalize()} Amount', color='white')
        ax.set_ylabel('Frequency', color='white')

    @safe_plot
    def plot_diagnosis_distribution(self, ax, data_type="claims"):
        """Plot diagnosis distribution."""
        if self.data_processor.data is None:
            return
            
        diagnosis_counts = self.data_processor.data['diagnosis'].value_counts().head(10)
        ax.barh(diagnosis_counts.index, diagnosis_counts.values, color='salmon')
        ax.set_xlabel(f'Number of {data_type.capitalize()}', color='white')
        ax.set_ylabel('Diagnosis', color='white')

    @safe_plot
    def plot_monthly_trend(self, ax, data_type="claims"):
        """Plot monthly trend."""
        if self.data_processor.data is None:
            return
            
        monthly_data = self.data_processor.data.groupby(
            self.data_processor.data['date'].dt.to_period('M')
        ).size()
        
        ax.plot(range(len(monthly_data)), monthly_data.values, 
                marker='o', color='lightblue')
        ax.set_xlabel('Month', color='white')
        ax.set_ylabel(f'Number of {data_type.capitalize()}', color='white')
        ax.set_xticks(range(len(monthly_data)))
        ax.set_xticklabels([str(p) for p in monthly_data.index], rotation=45)

    @safe_plot
    def plot_yearly_trend(self, ax, data_type="claims"):
        """Plot yearly trend."""
        if self.data_processor.data is None:
            return
            
        yearly_data = self.data_processor.data.groupby(
            self.data_processor.data['date'].dt.year
        ).agg({
            'amount': ['count', 'mean']
        })
        
        ax1 = ax
        ax2 = ax1.twinx()
        
        bars = ax1.bar(yearly_data.index, 
                      yearly_data[('amount', 'count')],
                      color='lightblue', alpha=0.7)
        line = ax2.plot(yearly_data.index,
                       yearly_data[('amount', 'mean')],
                       color='lightgreen', marker='o', linewidth=2)
        
        ax1.set_xlabel('Year', color='white')
        ax1.set_ylabel(f'Number of {data_type.capitalize()}', color='lightblue')
        ax2.set_ylabel('Average Amount', color='lightgreen')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', color='white')

    @safe_plot
    def plot_gender_distribution(self, ax, data_type="claims"):
        """Plot gender distribution."""
        if self.data_processor.data is None:
            return
            
        gender_age_stats = self.data_processor.data.groupby(
            ['gender', 'age_group']
        )['amount'].agg(['count', 'mean']).reset_index()
        
        bar_width = 0.35
        index = np.arange(len(gender_age_stats['age_group'].unique()))
        
        male_data = gender_age_stats[gender_age_stats['gender'] == 'M']
        female_data = gender_age_stats[gender_age_stats['gender'] == 'F']
        
        ax.bar(index - bar_width/2, male_data['count'],
               bar_width, label='Male', color='lightblue')
        ax.bar(index + bar_width/2, female_data['count'],
               bar_width, label='Female', color='lightpink')
        
        ax.set_xlabel('Age Group', color='white')
        ax.set_ylabel(f'Number of {data_type.capitalize()}', color='white')
        ax.set_xticks(index)
        ax.set_xticklabels(gender_age_stats['age_group'].unique(), rotation=45)
        ax.legend()

    @safe_plot
    def plot_correlation_heatmap(self, ax):
        """Plot correlation heatmap."""
        if self.data_processor.data is None:
            return
            
        numeric_cols = self.data_processor.data.select_dtypes(
            include=['int64', 'float64']
        ).columns
        corr_matrix = self.data_processor.data[numeric_cols].corr()
        
        sns.heatmap(corr_matrix,
                   annot=True,
                   cmap='coolwarm',
                   ax=ax,
                   fmt='.2f',
                   annot_kws={'size': 8},
                   mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
        
        ax.set_title('Correlation Heatmap', color='white', pad=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

    @safe_plot
    def plot_seasonal_pattern(self, ax, data_type="claims"):
        """Plot seasonal pattern."""
        if self.data_processor.data is None:
            return
            
        seasonal_stats = self.data_processor.data.groupby('season').agg({
            'amount': ['count', 'mean']
        }).reset_index()
        
        ax1 = ax
        ax2 = ax1.twinx()
        
        bars = ax1.bar(seasonal_stats['season'],
                      seasonal_stats[('amount', 'count')],
                      color='lightblue', alpha=0.7)
        line = ax2.plot(seasonal_stats['season'],
                       seasonal_stats[('amount', 'mean')],
                       color='lightgreen', marker='o', linewidth=2)
        
        ax1.set_xlabel('Season', color='white')
        ax1.set_ylabel(f'Number of {data_type.capitalize()}', color='lightblue')
        ax2.set_ylabel('Average Amount', color='lightgreen')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', color='white')
            

class GeneralPricingSystem(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.viz_manager = VisualizationManager(self.data_processor)
        self.logger = logging.getLogger(__name__)
        
        # Setup window
        self.setup_window()
        self.create_gui()
        
    def setup_window(self):
        """Initialize window settings."""
        self.title(Config.WINDOW_TITLE)
        self.geometry(Config.WINDOW_SIZE)
        
        ctk.set_appearance_mode(Config.APPEARANCE_MODE)
        ctk.set_default_color_theme(Config.COLOR_THEME)

    def create_gui(self):
        """Create the main GUI components."""
        self.create_main_container()
        self.create_header()
        self.create_status_label()
    
    def create_main_container(self):
        """Create the main container frame."""
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

    def create_header(self):
        """Create the header panel with logo."""
        # Header panel
        self.header_panel = ctk.CTkFrame(self.main_container, height=60, fg_color="white")
        self.header_panel.pack(fill="x", padx=5, pady=(5, 0))
        self.header_panel.pack_propagate(False)

        # Logo frame
        self.logo_frame = ctk.CTkFrame(self.header_panel, height=60, fg_color="white")
        self.logo_frame.pack(fill="x", padx=5, pady=2)
        self.logo_frame.pack_propagate(False)

        # Header content frame
        self.header_content = ctk.CTkFrame(self.logo_frame, height=50, fg_color="white")
        self.header_content.pack(fill="x", expand=True)

        # Compass label
        self.compass_label = ctk.CTkLabel(
            self.header_content,
            text="Compass",
            font=("Helvetica", 35, "bold"),
            text_color="black"
        )
        self.compass_label.pack(expand=True, pady=10)

        # Allianz Partners logo
        logo_image = Image.open("./allianz_partners_logo.png")
        ctk_logo = ctk.CTkImage(
            light_image=logo_image,
            dark_image=logo_image,
            size=(220, 40)
        )
        
        self.allianz_logo = ctk.CTkLabel(
            self.header_content,
            image=ctk_logo,
            text=""
        )
        self.allianz_logo.place(relx=1.0, rely=0.5, anchor="e", x=-20, y=0)

    def create_status_label(self):
        """Create status message display."""
        self.status_label = ctk.CTkLabel(
            self.main_container,
            text="",
            text_color="yellow"
        )
        self.status_label.pack(fill="x", padx=5)

        # Create notebook
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Setup all tabs
        self.setup_data_centre_tab()
        self.setup_performance_tab()
        self.setup_claim_tab()
        self.setup_correlation_tab()

    def setup_data_centre_tab(self):
        """Setup the Data Centre tab."""
        self.data_centre_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.data_centre_tab, text="Data Centre")

        # File upload frame
        self.upload_frame = ctk.CTkFrame(self.data_centre_tab)
        self.upload_frame.pack(fill="x", padx=20, pady=20)

        # Upload button
        self.upload_button = ctk.CTkButton(
            self.upload_frame,
            text="Upload Data File",
            command=self.upload_file,
            width=150
        )
        self.upload_button.pack(side="left", padx=5)

        # File path label
        self.file_label = ctk.CTkLabel(
            self.upload_frame,
            text="Please select a file",
            width=400
        )
        self.file_label.pack(side="left", padx=5, fill="x", expand=True)

        # Data preview frame
        self.preview_frame = ctk.CTkFrame(self.data_centre_tab)
        self.preview_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Preview label
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Data Preview",
            font=("Helvetica", 16, "bold")
        )
        self.preview_label.pack(pady=10)

        # Preview text widget
        self.preview_text = tk.Text(
            self.preview_frame,
            height=20,
            bg=Config.STYLES['bg_color'],
            fg=Config.STYLES['text_color'],
            font=('Courier', 10)
        )
        self.preview_text.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_performance_tab(self):
        """Setup the Performance tab."""
        self.performance_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.performance_tab, text="Performance")
        
        # Scrollable container
        self.performance_scrollable = ctk.CTkScrollableFrame(self.performance_tab)
        self.performance_scrollable.pack(fill="both", expand=True, padx=2, pady=2)

        # Graphs container
        self.performance_graphs = ctk.CTkFrame(self.performance_scrollable)
        self.performance_graphs.pack(fill="both", expand=True, padx=1, pady=1)

        # Create all performance graphs
        self.performance_frames = {
            'age': self.create_graph_frame(
                self.performance_graphs,
                "Performance by Age Group",
                row=0, column=0
            ),
            'amount': self.create_graph_frame(
                self.performance_graphs,
                "Performance Amount Distribution",
                row=0, column=1
            ),
            'diagnosis': self.create_graph_frame(
                self.performance_graphs,
                "Performance by Category",
                row=1, column=0
            ),
            'trend': self.create_graph_frame(
                self.performance_graphs,
                "Monthly Performance Trend",
                row=1, column=1
            ),
            'yearly': self.create_graph_frame(
                self.performance_graphs,
                "Yearly Performance Trend",
                row=2, column=0
            ),
            'avg_amount': self.create_graph_frame(
                self.performance_graphs,
                "Average Performance by Age",
                row=2, column=1
            ),
            'gender': self.create_graph_frame(
                self.performance_graphs,
                "Performance by Gender",
                row=3, column=0
            ),
            'seasonal': self.create_graph_frame(
                self.performance_graphs,
                "Seasonal Performance Pattern",
                row=3, column=1
            )
        }

        # Configure grid
        for i in range(4):
            self.performance_graphs.grid_columnconfigure(i, weight=1)
            self.performance_graphs.grid_rowconfigure(i, weight=1)

def setup_claim_tab(self):
        """Setup the Claim tab."""
        self.claim_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.claim_tab, text="Claim")

        # Scrollable container
        self.claim_scrollable = ctk.CTkScrollableFrame(self.claim_tab)
        self.claim_scrollable.pack(fill="both", expand=True, padx=2, pady=2)

        # Graphs container
        self.claim_graphs = ctk.CTkFrame(self.claim_scrollable)
        self.claim_graphs.pack(fill="both", expand=True, padx=1, pady=1)

        # Create all claim graphs
        self.claim_frames = {
            'age': self.create_graph_frame(
                self.claim_graphs,
                "Claims by Age Group",
                row=0, column=0
            ),
            'amount': self.create_graph_frame(
                self.claim_graphs,
                "Claim Amount Distribution",
                row=0, column=1
            ),
            'diagnosis': self.create_graph_frame(
                self.claim_graphs,
                "Claims by Diagnosis",
                row=1, column=0
            ),
            'trend': self.create_graph_frame(
                self.claim_graphs,
                "Monthly Claim Trend",
                row=1, column=1
            ),
            'yearly': self.create_graph_frame(
                self.claim_graphs,
                "Yearly Claim Trend",
                row=2, column=0
            ),
            'avg_amount': self.create_graph_frame(
                self.claim_graphs,
                "Average Claim by Age",
                row=2, column=1
            ),
            'gender': self.create_graph_frame(
                self.claim_graphs,
                "Claims by Gender",
                row=3, column=0
            ),
            'seasonal': self.create_graph_frame(
                self.claim_graphs,
                "Seasonal Pattern",
                row=3, column=1
            )
        }

        # Configure grid
        for i in range(4):
            self.claim_graphs.grid_columnconfigure(i, weight=1)
            self.claim_graphs.grid_rowconfigure(i, weight=1)

def setup_correlation_tab(self):
    """Setup the Correlation tab."""
    self.correlation_tab = ctk.CTkFrame(self.notebook)
    self.notebook.add(self.correlation_tab, text="Correlation")

    # Scrollable container
    self.correlation_scrollable = ctk.CTkScrollableFrame(self.correlation_tab)
    self.correlation_scrollable.pack(fill="both", expand=True, padx=5, pady=5)

    # Graphs container
    self.correlation_graphs = ctk.CTkFrame(self.correlation_scrollable)
    self.correlation_graphs.pack(fill="both", expand=True, padx=5, pady=5)

    # Create correlation graphs
    self.correlation_frames = {
        'heatmap': self.create_graph_frame(
            self.correlation_graphs,
            "Correlation Heatmap",
            row=0, column=0, columnspan=2
        ),
        'pairs': self.create_graph_frame(
            self.correlation_graphs,
            "Pairwise Relationships",
            row=1, column=0, columnspan=2
        ),
        'top': self.create_graph_frame(
            self.correlation_graphs,
            "Top 10 Correlations",
            row=2, column=0
        ),
        'scatter': self.create_graph_frame(
            self.correlation_graphs,
            "Key Variables vs Claims",
            row=2, column=1
        )
    }

    # Configure grid
    self.correlation_graphs.grid_columnconfigure(0, weight=1)
    self.correlation_graphs.grid_columnconfigure(1, weight=1)
    for i in range(3):
        self.correlation_graphs.grid_rowconfigure(i, weight=1)

def create_graph_frame(self, parent, title, row, column, columnspan=1):
    """Create a frame for graphs with consistent styling."""
    frame = ctk.CTkFrame(parent)
    frame.grid(row=row, column=column, sticky="nsew", 
              padx=5, pady=5, columnspan=columnspan)

    title_label = ctk.CTkLabel(
        frame,
        text=title,
        font=("Helvetica", 16, "bold")
    )
    title_label.pack(pady=5)

    fig = Figure(**Config.GRAPH_SETTINGS)
    ax = fig.add_subplot(111)
    ax.set_facecolor(Config.GRAPH_SETTINGS['facecolor'])
    
    # Configure axis styling
    ax.tick_params(colors=Config.STYLES['text_color'], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(Config.STYLES['text_color'])
        
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    return {"frame": frame, "fig": fig, "ax": ax, "canvas": canvas}

def upload_file(self):
        """Handle file upload process."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("Excel files (old)", "*.xls"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                self.status_label.configure(text="File selection cancelled")
                return

            file_path = Path(file_path)
            self.file_label.configure(text=str(file_path))
            
            if file_path.suffix.lower() not in ['.csv', '.xlsx', '.xls']:
                raise ValueError("Unsupported file format")

            self.status_label.configure(text="Loading file...")
            self.load_and_process_data(file_path)
            
        except Exception as e:
            self.handle_error("Error during file upload", e)

def load_and_process_data(self, file_path: Path):
  """Load and process the uploaded data file."""
  try:
      self.data_processor.load_data(file_path)
      self.update_all_visualizations()
      self.status_label.configure(
          text=f"Data loaded successfully: {len(self.data_processor.data)} records"
      )
  except Exception as e:
      self.handle_error("Error processing data", e)

def update_all_visualizations(self):
    """Update all visualizations after data load."""
    self.update_data_preview()
    self.update_performance_plots()
    self.update_claim_plots()
    self.update_correlation_plots()

def update_data_preview(self):
    """Update the data preview text."""
    if self.data_processor.data is None:
        return
        
    self.preview_text.delete(1.0, tk.END)
    self.preview_


def update_data_preview(self):
    """Update the data preview text."""
    if self.data_processor.data is None:
        return
        
    self.preview_text.delete(1.0, tk.END)
    self.preview_text.insert(1.0, self.data_processor.get_data_summary())

def update_performance_plots(self):
    """Update all performance plots."""
    if self.data_processor.data is None:
        return

    plot_functions = {
        'age': lambda frame: self.viz_manager.plot_age_distribution(
            frame["ax"], "performance"),
        'amount': lambda frame: self.viz_manager.plot_amount_distribution(
            frame["ax"], "performance"),
        'diagnosis': lambda frame: self.viz_manager.plot_diagnosis_distribution(
            frame["ax"], "performance"),
        'trend': lambda frame: self.viz_manager.plot_monthly_trend(
            frame["ax"], "performance"),
        'yearly': lambda frame: self.viz_manager.plot_yearly_trend(
            frame["ax"], "performance"),
        'gender': lambda frame: self.viz_manager.plot_gender_distribution(
            frame["ax"], "performance"),
        'seasonal': lambda frame: self.viz_manager.plot_seasonal_pattern(
            frame["ax"], "performance")
    }

    for plot_type, plot_func in plot_functions.items():
        if plot_type in self.performance_frames:
            frame = self.performance_frames[plot_type]
            frame["ax"].clear()
            plot_func(frame)
            frame["canvas"].draw()

def update_claim_plots(self):
    """Update all claim plots."""
    if self.data_processor.data is None:
        return

    plot_functions = {
        'age': lambda frame: self.viz_manager.plot_age_distribution(
            frame["ax"], "claims"),
        'amount': lambda frame: self.viz_manager.plot_amount_distribution(
            frame["ax"], "claims"),
        'diagnosis': lambda frame: self.viz_manager.plot_diagnosis_distribution(
            frame["ax"], "claims"),
        'trend': lambda frame: self.viz_manager.plot_monthly_trend(
            frame["ax"], "claims"),
        'yearly': lambda frame: self.viz_manager.plot_yearly_trend(
            frame["ax"], "claims"),
        'gender': lambda frame: self.viz_manager.plot_gender_distribution(
            frame["ax"], "claims"),
        'seasonal': lambda frame: self.viz_manager.plot_seasonal_pattern(
            frame["ax"], "claims")
    }

    for plot_type, plot_func in plot_functions.items():
        if plot_type in self.claim_frames:
            frame = self.claim_frames[plot_type]
            frame["ax"].clear()
            plot_func(frame)
            frame["canvas"].draw()

def update_correlation_plots(self):
    """Update all correlation plots."""
    if self.data_processor.data is None:
        return

    # Update correlation heatmap
    if 'heatmap' in self.correlation_frames:
        frame = self.correlation_frames['heatmap']
        frame["ax"].clear()
        self.viz_manager.plot_correlation_heatmap(frame["ax"])
        frame["canvas"].draw()

    # Update other correlation plots
    numeric_cols = self.data_processor.data.select_dtypes(
        include=['int64', 'float64']
    ).columns
    
    if 'pairs' in self.correlation_frames:
        frame = self.correlation_frames['pairs']
        frame["ax"].clear()
        
        corr_matrix = self.data_processor.data[numeric_cols].corr()
        top_features = abs(corr_matrix['amount']).sort_values(ascending=False)[:4].index
        
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features):
                plt_ax = frame["ax"].inset_axes([0.25*i, 0.25*j, 0.23, 0.23])
                if i != j:
                    plt_ax.scatter(
                        self.data_processor.data[feat1],
                        self.data_processor.data[feat2],
                        alpha=0.5,
                        c='lightblue',
                        s=20
                    )
                else:
                    plt_ax.hist(
                        self.data_processor.data[feat1],
                        bins=20,
                        color='lightgreen'
                    )
                if i == 0:
                    plt_ax.set_ylabel(feat2, color='white')
                if j == len(top_features)-1:
                    plt_ax.set_xlabel(feat1, color='white')
                plt_ax.tick_params(colors='white', labelsize=8)
        
        frame["ax"].set_title('Pairwise Relationships of Top Features',
                            color='white', pad=20)
        frame["canvas"].draw()

    if 'top' in self.correlation_frames:
        frame = self.correlation_frames['top']
        frame["ax"].clear()
        
        corr_matrix = self.data_processor.data[numeric_cols].corr()
        top_corr = corr_matrix['amount'].sort_values(ascending=True)
        top_corr = top_corr.drop('amount')
        
        colors = ['red' if x < 0 else 'green' for x in top_corr]
        bars = frame["ax"].barh(range(len(top_corr)), top_corr, color=colors)
        
        frame["ax"].set_yticks(range(len(top_corr)))
        frame["ax"].set_yticklabels(top_corr.index, fontsize=8)
        frame["ax"].set_xlabel('Correlation Coefficient', color='white')
        frame["ax"].set_title('Correlations with Claim Amount',
                            color='white', pad=20)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            frame["ax"].text(
                width, i,
                f'{width:.2f}',
                color='white',
                va='center',
                ha='left' if width >= 0 else 'right'
            )
        
        frame["canvas"].draw()

    if 'scatter' in self.correlation_frames:
        frame = self.correlation_frames['scatter']
        frame["ax"].clear()
        
        corr_matrix = self.data_processor.data[numeric_cols].corr()
        top_vars = abs(corr_matrix['amount']).sort_values(ascending=False)[1:4].index
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_vars)))
        for var, color in zip(top_vars, colors):
            frame["ax"].scatter(
                self.data_processor.data[var],
                self.data_processor.data['amount'],
                alpha=0.5,
                c=[color],
                label=var
            )
        
        frame["ax"].set_xlabel('Feature Value', color='white')
        frame["ax"].set_ylabel('Claim Amount', color='white')
        frame["ax"].legend()
        frame["ax"].set_title('Top Correlating Variables vs Claims',
                            color='white', pad=20)
        
        frame["canvas"].draw()

def handle_error(self, message: str, error: Exception):
    """Centralized error handling."""
    error_msg = f"{message}: {str(error)}"
    self.status_label.configure(text=error_msg)
    messagebox.showerror("Error", error_msg)
    self.logger.error(f"{message}:\n{traceback.format_exc()}")

if __name__ == "__main__":
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        app = GeneralPricingSystem()
        app.mainloop()
    except Exception as e:
        logging.error(f"Application error: {str(e)}\n{traceback.format_exc()}")