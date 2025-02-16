import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
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

        # Scrollable container with reduced padding
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.main_container
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Graph container with compact layout
        self.graphs_container = ctk.CTkFrame(self.scrollable_frame)
        self.graphs_container.pack(fill="both", expand=True, padx=1, pady=1)

        # Graph frames with tighter spacing
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

        # Tighter grid settings
        for i in range(4):
            self.graphs_container.grid_rowconfigure(i, weight=1, minsize=200)  # Reduce minimum size
        for i in range(2):
            self.graphs_container.grid_columnconfigure(i, weight=1)

    def create_graph_frame(self, parent, title, row, column):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)

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
                    text=f"Warning: Missing values found - {dict(na_counts[na_counts > 0])}")
            
            self.update_visualizations()
            self.status_label.configure(text=f"Data loaded successfully: {len(self.data)} records")

        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            self.status_label.configure(text=error_msg)
            messagebox.showerror("Error", error_msg)
            print("Detailed error information:")
            print(traceback.format_exc())

    def update_visualizations(self):
        if self.data is None:
            return

        # Update all graphs
        self.update_age_distribution()
        self.update_amount_distribution()
        self.update_diagnosis_distribution()
        self.update_monthly_trend()
        self.update_yearly_trend()
        self.update_avg_amount_by_age()
        self.update_gender_distribution()
        self.update_seasonal_pattern()

    def update_age_distribution(self):
        ax = self.age_frame["ax"]
        canvas = self.age_frame["canvas"]
        
        ax.clear()
        # Age group settings and visualization
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
        
        if 'age' in self.data.columns:
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            age_dist = self.data['age_group'].value_counts()
            
            ax.bar(age_dist.index, age_dist.values, color='skyblue')
            ax.set_xlabel('Age Group', color='white', fontsize=10, fontweight='bold')
            ax.set_ylabel('Number of Claims', color='white', fontsize=10, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)
            
            # Add title with smaller font
            ax.set_title('Distribution of Claims by Age', color='white', fontsize=12, pad=10)
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_amount_distribution(self):
        ax = self.amount_frame["ax"]
        canvas = self.amount_frame["canvas"]
        
        ax.clear()
        if 'amount' in self.data.columns:
            ax.hist(self.data['amount'], bins=50, color='lightgreen')
            ax.set_xlabel('Claim Amount', color='white')
            ax.set_ylabel('Frequency', color='white')
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_diagnosis_distribution(self):
        ax = self.diagnosis_frame["ax"]
        canvas = self.diagnosis_frame["canvas"]
        
        ax.clear()
        if 'diagnosis' in self.data.columns:
            diagnosis_counts = self.data['diagnosis'].value_counts().head(10)
            ax.barh(diagnosis_counts.index, diagnosis_counts.values, color='salmon')
            ax.set_xlabel('Number of Claims', color='white')
            ax.set_ylabel('Diagnosis', color='white')
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_monthly_trend(self):
        ax = self.trend_frame["ax"]
        canvas = self.trend_frame["canvas"]
        
        ax.clear()
        if 'date' in self.data.columns:
            monthly_claims = self.data.groupby(self.data['date'].dt.to_period('M')).size()
            
            ax.plot(range(len(monthly_claims)), monthly_claims.values, 
                   marker='o', color='lightblue', linewidth=2)
            ax.set_xlabel('Month', color='white', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Claims', color='white', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(monthly_claims)))
            ax.set_xticklabels([str(p) for p in monthly_claims.index], rotation=45, fontsize=10)
            ax.set_title('Monthly Trend of Claims', color='white', fontsize=14, pad=20)
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_yearly_trend(self):
        ax = self.yearly_trend_frame["ax"]
        canvas = self.yearly_trend_frame["canvas"]
        
        ax.clear()
        if 'date' in self.data.columns:
            yearly_claims = self.data.groupby(self.data['date'].dt.year).agg({
                'amount': ['count', 'sum', 'mean']
            })
            
            # Plot multiple metrics
            years = yearly_claims.index
            claims_count = yearly_claims[('amount', 'count')]
            claims_avg = yearly_claims[('amount', 'mean')]
            
            color1, color2 = 'lightblue', 'lightgreen'
            ax1 = ax
            ax2 = ax1.twinx()
            
            # Plot number of claims
            line1 = ax1.bar(years, claims_count, color=color1, alpha=0.7)
            ax1.set_xlabel('Year', color='white', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Claims', color=color1, fontsize=12, fontweight='bold')
            
            # Plot average claim amount
            line2 = ax2.plot(years, claims_avg, color=color2, linewidth=3, marker='o')
            ax2.set_ylabel('Average Claim Amount', color=color2, fontsize=12, fontweight='bold')
            
            # Customize appearance
            ax1.tick_params(axis='y', labelcolor=color1)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax.set_title('Yearly Claims Trend', color='white', fontsize=14, pad=20)
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_avg_amount_by_age(self):
        ax = self.avg_amount_frame["ax"]
        canvas = self.avg_amount_frame["canvas"]
        
        ax.clear()
        if all(col in self.data.columns for col in ['age', 'amount']):
            # Create age groups and calculate average amount
            age_bins = [0, 18, 30, 45, 60, 75, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
            
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            avg_by_age = self.data.groupby('age_group')['amount'].mean()
            
            # Create bar plot with gradient colors
            bars = ax.bar(avg_by_age.index, avg_by_age.values)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', color='white', fontsize=10)
            
            ax.set_xlabel('Age Group', color='white', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Claim Amount ($)', color='white', fontsize=12, fontweight='bold')
            ax.set_title('Average Claim Amount by Age Group', color='white', fontsize=14, pad=20)
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_gender_distribution(self):
        ax = self.gender_dist_frame["ax"]
        canvas = self.gender_dist_frame["canvas"]
        
        ax.clear()
        if 'gender' in self.data.columns:
            gender_stats = self.data.groupby(['gender', pd.cut(self.data['age'], 
                                            bins=[0, 18, 30, 45, 60, 75, 100],
                                            labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
                                            )])['amount'].agg(['count', 'mean']).reset_index()
            
            # Plot grouped bar chart
            bar_width = 0.35
            index = np.arange(len(gender_stats['age'].unique()))
            
            ax.bar(index, gender_stats[gender_stats['gender'] == 'M']['count'], 
                  bar_width, label='Male', color='lightblue')
            ax.bar(index + bar_width, gender_stats[gender_stats['gender'] == 'F']['count'], 
                  bar_width, label='Female', color='lightpink')
            
            ax.set_xlabel('Age Group', color='white', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Claims', color='white', fontsize=12, fontweight='bold')
            ax.set_title('Claims Distribution by Gender and Age', color='white', fontsize=14, pad=20)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(gender_stats['age'].unique(), rotation=45, fontsize=10)
            ax.legend()
            
        else:
            ax.text(0.5, 0.5, 'Gender data not available', 
                   ha='center', va='center', color='white', fontsize=12)
        
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

    def update_seasonal_pattern(self):
        ax = self.seasonal_frame["ax"]
        canvas = self.seasonal_frame["canvas"]
        
        ax.clear()
        if 'date' in self.data.columns:
            # Extract month and season
            self.data['month'] = self.data['date'].dt.month
            self.data['season'] = pd.cut(self.data['date'].dt.month, 
                                       bins=[0, 3, 6, 9, 12], 
                                       labels=['Winter', 'Spring', 'Summer', 'Fall'])
            
            # Calculate seasonal averages
            seasonal_stats = self.data.groupby('season').agg({
                'amount': ['count', 'mean']
            }).reset_index()
            
            # Create two subplots
            ax.clear()
            ax1 = ax
            ax2 = ax1.twinx()
            
            # Plot bars for claim counts
            bars = ax1.bar(seasonal_stats['season'], 
                          seasonal_stats[('amount', 'count')],
                          color='lightblue', alpha=0.7)
            ax1.set_ylabel('Number of Claims', color='lightblue', fontsize=12, fontweight='bold')
            
            # Plot line for average amount
            line = ax2.plot(seasonal_stats['season'], 
                          seasonal_stats[('amount', 'mean')],
                          color='lightgreen', marker='o', linewidth=3)
            ax2.set_ylabel('Average Claim Amount ($)', color='lightgreen', 
                          fontsize=12, fontweight='bold')
            
            # Customize appearance
            ax.set_title('Seasonal Claim Patterns', color='white', fontsize=14, pad=20)
            ax1.set_xlabel('Season', color='white', fontsize=12, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white', fontsize=10)
            
        ax.grid(True, color='gray', alpha=0.3)
        canvas.draw()

if __name__ == "__main__":
    try:
        app = InsuranceClaimAnalyzer()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())