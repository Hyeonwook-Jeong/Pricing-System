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
from pathlib import Path

class InsuranceClaimAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window settings
        self.title("Insurance Claim Analysis System")
        self.geometry("1200x800")
        
        # Theme settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.data = None
        self.create_gui()

    def create_gui(self):
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Control panel
        self.control_panel = ctk.CTkFrame(self.main_container)
        self.control_panel.pack(fill="x", padx=5, pady=5)

        # File upload frame
        self.upload_frame = ctk.CTkFrame(self.control_panel)
        self.upload_frame.pack(fill="x", padx=5, pady=5)

        # File upload button
        self.upload_button = ctk.CTkButton(
            self.upload_frame,
            text="Upload Data File",
            command=self.upload_file,
            width=150
        )
        self.upload_button.pack(side="left", padx=5)

        # File path display
        self.file_label = ctk.CTkLabel(
            self.upload_frame,
            text="Please select a file",
            width=400
        )
        self.file_label.pack(side="left", padx=5, fill="x", expand=True)

        # Status message display
        self.status_label = ctk.CTkLabel(
            self.control_panel,
            text="",
            text_color="yellow"
        )
        self.status_label.pack(fill="x", padx=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create claim trend tab
        self.main_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.main_tab, text="Claim Trend")

        # Create correlation tab
        self.correlation_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.correlation_tab, text="Correlation")

        # Setup main analysis tab
        self.setup_main_analysis_tab()
        
        # Setup correlation analysis tab
        self.setup_correlation_analysis_tab()

    def setup_main_analysis_tab(self):
        # Scrollable container
        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_tab)
        self.scrollable_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Graph container
        self.graphs_container = ctk.CTkFrame(self.scrollable_frame)
        self.graphs_container.pack(fill="both", expand=True, padx=1, pady=1)

        # Graph frames
        self.age_frame = self.create_graph_frame(
            self.graphs_container,
            "Claims by Age Group",
            row=0, column=0
        )

        self.amount_frame = self.create_graph_frame(
            self.graphs_container,
            "Claim Amount Distribution",
            row=0, column=1
        )

        self.diagnosis_frame = self.create_graph_frame(
            self.graphs_container,
            "Claims by Diagnosis",
            row=1, column=0
        )

        self.trend_frame = self.create_graph_frame(
            self.graphs_container,
            "Monthly Claim Trend",
            row=1, column=1
        )

        self.yearly_trend_frame = self.create_graph_frame(
            self.graphs_container,
            "Yearly Claim Trend",
            row=2, column=0
        )

        self.avg_amount_frame = self.create_graph_frame(
            self.graphs_container,
            "Average Claim by Age",
            row=2, column=1
        )

        self.gender_dist_frame = self.create_graph_frame(
            self.graphs_container,
            "Claims by Gender",
            row=3, column=0
        )

        self.seasonal_frame = self.create_graph_frame(
            self.graphs_container,
            "Seasonal Pattern",
            row=3, column=1
        )

        # Configure grid
        for i in range(4):  # Updated to 4 rows
            self.graphs_container.grid_columnconfigure(i, weight=1)
            self.graphs_container.grid_rowconfigure(i, weight=1)

    def setup_correlation_analysis_tab(self):
        # Correlation visualization frames
        self.correlation_frames = ctk.CTkFrame(self.correlation_tab)
        self.correlation_frames.pack(fill="both", expand=True, padx=5, pady=5)

        # Create frames for correlation analysis
        self.correlation_heatmap_frame = self.create_graph_frame(
            self.correlation_frames,
            "Claims Correlation Heatmap",
            row=0, column=0
        )

        self.top_correlations_frame = self.create_graph_frame(
            self.correlation_frames,
            "Top Correlations with Claims",
            row=0, column=1
        )

        self.correlation_scatter_frame = self.create_graph_frame(
            self.correlation_frames,
            "Key Variables vs Claims",
            row=1, column=0, columnspan=2
        )

        # Configure grid
        self.correlation_frames.grid_columnconfigure(0, weight=1)
        self.correlation_frames.grid_columnconfigure(1, weight=1)
        self.correlation_frames.grid_rowconfigure(0, weight=1)
        self.correlation_frames.grid_rowconfigure(1, weight=1)

    def update_correlation_plots(self):
        if self.data is None:
            return

        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()

        # Update correlation heatmap
        ax = self.correlation_heatmap_frame["ax"]
        ax.clear()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Claims Correlation Heatmap', color='white', pad=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        self.correlation_heatmap_frame["canvas"].draw()

        # Update top correlations with claims
        ax = self.top_correlations_frame["ax"]
        ax.clear()
        
        # Get correlations with 'amount' and sort
        claim_correlations = corr_matrix['amount'].sort_values(ascending=True)
        claim_correlations = claim_correlations.drop('amount')  # Remove self-correlation
        
        # Plot horizontal bar chart of correlations
        colors = ['red' if x < 0 else 'green' for x in claim_correlations]
        ax.barh(range(len(claim_correlations)), claim_correlations, color=colors)
        ax.set_yticks(range(len(claim_correlations)))
        ax.set_yticklabels(claim_correlations.index, fontsize=10)
        ax.set_xlabel('Correlation Coefficient', color='white')
        ax.set_title('Top Correlations with Claim Amount', color='white', pad=20)
        
        # Add correlation values as text
        for i, v in enumerate(claim_correlations):
            ax.text(v, i, f'{v:.2f}', color='white', va='center')
            
        self.top_correlations_frame["canvas"].draw()

        # Update scatter plots for top correlating variables
        ax = self.correlation_scatter_frame["ax"]
        ax.clear()
        
        # Get top 3 most correlated variables (absolute value)
        top_vars = claim_correlations.abs().nlargest(3).index
        
        # Create subplots
        for i, var in enumerate(top_vars, 1):
            plt_ax = ax.inset_axes([0.3 * (i-1), 0.1, 0.25, 0.8])
            plt_ax.scatter(self.data[var], self.data['amount'], alpha=0.5, c='lightblue')
            plt_ax.set_xlabel(var, color='white')
            if i == 1:
                plt_ax.set_ylabel('Claim Amount', color='white')
            plt_ax.tick_params(colors='white')
            
        ax.set_title('Key Variables vs Claim Amount', color='white', pad=20)
        self.correlation_scatter_frame["canvas"].draw()

    def update_main_visualizations(self):
        if self.data is None:
            return

        # Update age distribution
        ax = self.age_frame["ax"]
        ax.clear()
        if 'age' in self.data.columns:
            age_bins = [0, 18, 30, 45, 60, 75, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            age_dist = self.data['age_group'].value_counts()
            ax.bar(age_dist.index, age_dist.values, color='skyblue')
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Number of Claims', color='white')
            plt.setp(ax.get_xticklabels(), rotation=45)
        self.age_frame["canvas"].draw()

        # Update amount distribution
        ax = self.amount_frame["ax"]
        ax.clear()
        if 'amount' in self.data.columns:
            ax.hist(self.data['amount'], bins=50, color='lightgreen')
            ax.set_xlabel('Claim Amount', color='white')
            ax.set_ylabel('Frequency', color='white')
        self.amount_frame["canvas"].draw()

        # Update diagnosis distribution
        ax = self.diagnosis_frame["ax"]
        ax.clear()
        if 'diagnosis' in self.data.columns:
            diagnosis_counts = self.data['diagnosis'].value_counts().head(10)
            ax.barh(diagnosis_counts.index, diagnosis_counts.values, color='salmon')
            ax.set_xlabel('Number of Claims', color='white')
            ax.set_ylabel('Diagnosis', color='white')
        self.diagnosis_frame["canvas"].draw()

        # Update monthly trend
        ax = self.trend_frame["ax"]
        ax.clear()
        if 'date' in self.data.columns:
            monthly_claims = self.data.groupby(self.data['date'].dt.to_period('M')).size()
            ax.plot(range(len(monthly_claims)), monthly_claims.values, marker='o')
            ax.set_xlabel('Month', color='white')
            ax.set_ylabel('Number of Claims', color='white')
            ax.set_xticks(range(len(monthly_claims)))
            ax.set_xticklabels([str(p) for p in monthly_claims.index], rotation=45)
        self.trend_frame["canvas"].draw()

        # Update yearly trend
        ax = self.yearly_trend_frame["ax"]
        ax.clear()
        if 'date' in self.data.columns:
            yearly_data = self.data.groupby(self.data['date'].dt.year).agg({
                'amount': ['count', 'mean']
            })
            
            # Plot bars for number of claims
            ax1 = ax
            ax2 = ax1.twinx()
            
            bars = ax1.bar(yearly_data.index, yearly_data[('amount', 'count')], color='lightblue', alpha=0.7)
            line = ax2.plot(yearly_data.index, yearly_data[('amount', 'mean')], color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Year', color='white')
            ax1.set_ylabel('Number of Claims', color='lightblue')
            ax2.set_ylabel('Average Claim Amount', color='lightgreen')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        self.yearly_trend_frame["canvas"].draw()

        # Update average claim by age
        ax = self.avg_amount_frame["ax"]
        ax.clear()
        if all(col in self.data.columns for col in ['age', 'amount']):
            age_bins = [0, 18, 30, 45, 60, 75, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            avg_by_age = self.data.groupby('age_group')['amount'].mean()
            
            bars = ax.bar(avg_by_age.index, avg_by_age.values, color='lightblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', color='white')
                
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Average Claim Amount ($)', color='white')
            plt.setp(ax.get_xticklabels(), rotation=45)
            
        self.avg_amount_frame["canvas"].draw()

        # Update gender distribution
        ax = self.gender_dist_frame["ax"]
        ax.clear()
        if 'gender' in self.data.columns:
            gender_age_stats = self.data.groupby(['gender', pd.cut(self.data['age'], 
                                                bins=[0, 18, 30, 45, 60, 75, 100],
                                                labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
                                                )])['amount'].agg(['count', 'mean']).reset_index()
            
            bar_width = 0.35
            index = np.arange(len(gender_age_stats['age'].unique()))
            
            # Plot bars for males
            male_data = gender_age_stats[gender_age_stats['gender'] == 'M']
            female_data = gender_age_stats[gender_age_stats['gender'] == 'F']
            
            ax.bar(index - bar_width/2, male_data['count'], bar_width, label='Male', color='lightblue')
            ax.bar(index + bar_width/2, female_data['count'], bar_width, label='Female', color='lightpink')
            
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Number of Claims', color='white')
            ax.set_xticks(index)
            ax.set_xticklabels(gender_age_stats['age'].unique(), rotation=45)
            ax.legend()
            
        self.gender_dist_frame["canvas"].draw()

        # Update seasonal pattern
        ax = self.seasonal_frame["ax"]
        ax.clear()
        if 'date' in self.data.columns:
            self.data['month'] = self.data['date'].dt.month
            self.data['season'] = pd.cut(self.data['date'].dt.month, 
                                       bins=[0, 3, 6, 9, 12], 
                                       labels=['Winter', 'Spring', 'Summer', 'Fall'])
            
            seasonal_stats = self.data.groupby('season').agg({
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
            ax1.set_ylabel('Number of Claims', color='lightblue')
            ax2.set_ylabel('Average Claim Amount', color='lightgreen')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        self.seasonal_frame["canvas"].draw()

    def create_graph_frame(self, parent, title, row, column, columnspan=1):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=column, sticky="nsew", padx=5, pady=5, columnspan=columnspan)

        # Title
        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=5)

        # matplotlib graph with improved visibility
        fig = Figure(figsize=(8, 6), dpi=100)
        fig.patch.set_facecolor('#2B2B2B')
        
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2B2B2B')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')
            
        # Increase spacing for labels
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
            
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        return {"frame": frame, "fig": fig, "ax": ax, "canvas": canvas}

    def upload_file(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("CSV files", ".csv"),
                    ("Excel files", ".xlsx"),
                    ("Excel files (old)", ".xls"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:  # File selection cancelled
                self.status_label.configure(text="File selection cancelled")
                return

            file_path = Path(file_path)
            self.file_label.configure(text=str(file_path))
            
            # Check file extension
            if file_path.suffix.lower() not in ['.csv', '.xlsx', '.xls']:
                raise ValueError("Unsupported file format")

            self.status_label.configure(text="Loading file...")
            self.load_and_process_data(file_path)
            
        except Exception as e:
            error_msg = f"Error during file upload: {str(e)}"
            self.status_label.configure(text=error_msg)
            messagebox.showerror("Error", error_msg)
            print("Detailed error information:")
            print(traceback.format_exc())

    def load_and_process_data(self, file_path):
        try:
            # Load data based on file format
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)

            # Check required columns
            required_columns = ['age', 'amount', 'diagnosis', 'date']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            # Basic data preprocessing
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')
            self.data['amount'] = pd.to_numeric(self.data['amount'], errors='coerce')

            # Check for missing values
            na_counts = self.data[required_columns].isna().sum()
            if na_counts.any():
                self.status_label.configure(
                    text=f"Warning: Missing values found - {dict(na_counts[na_counts > 0])}"
                )
            
            # Update all visualizations
            self.update_main_visualizations()
            self.update_correlation_plots()
            
            self.status_label.configure(text=f"Data loaded successfully: {len(self.data)} records")

        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            self.status_label.configure(text=error_msg)
            messagebox.showerror("Error", error_msg)
            print("Detailed error information:")
            print(traceback.format_exc())

if __name__ == "__main__":
    try:
        app = InsuranceClaimAnalyzer()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())