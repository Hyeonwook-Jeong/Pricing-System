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
from PIL import Image, ImageTk

class GeneralPricingSystem(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window settings
        self.title("GPS")
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

        # Header panel (for logo and program name)
        self.header_panel = ctk.CTkFrame(self.main_container, height=60, fg_color="white")
        self.header_panel.pack(fill="x", padx=5, pady=(5, 0))
        self.header_panel.pack_propagate(False)

        # Logo frame with Compass text and Allianz logo
        self.logo_frame = ctk.CTkFrame(self.header_panel, height=60, fg_color="white")
        self.logo_frame.pack(fill="x", padx=5, pady=2)
        self.logo_frame.pack_propagate(False)

        # Create a frame for organizing horizontal layout
        self.header_content = ctk.CTkFrame(self.logo_frame, height=50, fg_color="white")
        self.header_content.pack(fill="x", expand=True)

        # Add Compass text in the center
        self.compass_label = ctk.CTkLabel(
            self.header_content,
            text="Compass",
            font=("Helvetica", 35, "bold"),
            text_color="black"
        )
        self.compass_label.pack(expand=True, pady=10)

        # Load and resize Allianz Partners logo
        logo_image = Image.open("allianz_partners_logo.png")
        ctk_logo = ctk.CTkImage(
            light_image=logo_image,
            dark_image=logo_image,
            size=(220, 40)
        )
        
        # Add Allianz Partners logo
        self.allianz_logo = ctk.CTkLabel(
            self.header_content,
            image=ctk_logo,
            text=""
        )
        self.allianz_logo.place(relx=1.0, rely=0.5, anchor="e", x=-20, y=0)
        
        # Status message display
        self.status_label = ctk.CTkLabel(
            self.main_container,
            text="",
            text_color="yellow"
        )
        self.status_label.pack(fill="x", padx=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create Data Centre tab
        self.data_centre_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.data_centre_tab, text="Data Centre")

        # Create performance tab
        self.performance_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.performance_tab, text="Performance")

        # Create claim trend tab
        self.claim_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.claim_tab, text="Claim")

        # Create correlation tab
        self.correlation_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.correlation_tab, text="Correlation")

        # Setup all tabs
        self.setup_data_centre_tab()
        self.setup_performance_tab()
        self.setup_claim_tab()
        self.setup_correlation_tab()

    def setup_data_centre_tab(self):
        # File upload frame
        self.upload_frame = ctk.CTkFrame(self.data_centre_tab)
        self.upload_frame.pack(fill="x", padx=20, pady=20)

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

        # Data preview frame
        self.preview_frame = ctk.CTkFrame(self.data_centre_tab)
        self.preview_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Add data preview label
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Data Preview",
            font=("Helvetica", 16, "bold")
        )
        self.preview_label.pack(pady=10)

        # Create text widget for data preview
        self.preview_text = tk.Text(
            self.preview_frame,
            height=20,
            bg='#2B2B2B',
            fg='white',
            font=('Courier', 10)
        )
        self.preview_text.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_performance_tab(self):
        # Scrollable container
        self.performance_scrollable_frame = ctk.CTkScrollableFrame(self.performance_tab)
        self.performance_scrollable_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Graph container
        self.performance_graphs_container = ctk.CTkFrame(self.performance_scrollable_frame)
        self.performance_graphs_container.pack(fill="both", expand=True, padx=1, pady=1)

        # Graph frames for performance visualization
        self.performance_age_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Performance by Age Group",
            row=0, column=0
        )

        self.performance_amount_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Performance Amount Distribution",
            row=0, column=1
        )

        self.performance_diagnosis_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Performance by Category",
            row=1, column=0
        )

        self.performance_trend_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Monthly Performance Trend",
            row=1, column=1
        )

        self.performance_yearly_trend_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Yearly Performance Trend",
            row=2, column=0
        )

        self.performance_avg_amount_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Average Performance by Age",
            row=2, column=1
        )

        self.performance_gender_dist_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Performance by Gender",
            row=3, column=0
        )

        self.performance_seasonal_frame = self.create_graph_frame(
            self.performance_graphs_container,
            "Seasonal Performance Pattern",
            row=3, column=1
        )

        # Configure grid for performance tab
        for i in range(4):
            self.performance_graphs_container.grid_columnconfigure(i, weight=1)
            self.performance_graphs_container.grid_rowconfigure(i, weight=1)

    def setup_claim_tab(self):
        # Scrollable container
        self.scrollable_frame = ctk.CTkScrollableFrame(self.claim_tab)
        self.scrollable_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Graph container
        self.graphs_container = ctk.CTkFrame(self.scrollable_frame)
        self.graphs_container.pack(fill="both", expand=True, padx=1, pady=1)

        # Graph frames for claim visualization
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

        # Configure grid for claim tab
        for i in range(4):
            self.graphs_container.grid_columnconfigure(i, weight=1)
            self.graphs_container.grid_rowconfigure(i, weight=1)

    def setup_correlation_tab(self):
        # Create scrollable frame
        self.correlation_scroll = ctk.CTkScrollableFrame(self.correlation_tab)
        self.correlation_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Correlation visualization frames
        self.correlation_frames = ctk.CTkFrame(self.correlation_scroll)
        self.correlation_frames.pack(fill="both", expand=True, padx=5, pady=5)

        # Create frames for correlation analysis
        self.correlation_heatmap_frame = self.create_graph_frame(
            self.correlation_frames,
            "Correlation Heatmap",
            row=0, column=0, columnspan=2
        )

        self.correlation_pairs_frame = self.create_graph_frame(
            self.correlation_frames,
            "Pairwise Relationships",
            row=1, column=0, columnspan=2
        )

        self.top_correlations_frame = self.create_graph_frame(
            self.correlation_frames,
            "Top 10 Correlations",
            row=2, column=0
        )

        self.correlation_scatter_frame = self.create_graph_frame(
            self.correlation_frames,
            "Key Variables vs Claims",
            row=2, column=1
        )

        # Configure grid for correlation tab
        self.correlation_frames.grid_columnconfigure(0, weight=1)
        self.correlation_frames.grid_columnconfigure(1, weight=1)
        for i in range(3):
            self.correlation_frames.grid_rowconfigure(i, weight=1)

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
            
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
            
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        return {"frame": frame, "fig": fig, "ax": ax, "canvas": canvas}

    def update_performance_plots(self):
        if self.data is None:
            return

        # Update age distribution
        ax = self.performance_age_frame["ax"]
        ax.clear()
        if 'age' in self.data.columns:
            age_bins = [0, 18, 30, 45, 60, 75, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            age_dist = self.data['age_group'].value_counts()
            ax.bar(age_dist.index, age_dist.values, color='skyblue')
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Performance Count', color='white')
            plt.setp(ax.get_xticklabels(), rotation=45)
        self.performance_age_frame["canvas"].draw()

        # Update amount distribution
        ax = self.performance_amount_frame["ax"]
        ax.clear()
        if 'amount' in self.data.columns:
            ax.hist(self.data['amount'], bins=50, color='lightgreen')
            ax.set_xlabel('Performance Amount', color='white')
            ax.set_ylabel('Frequency', color='white')
        self.performance_amount_frame["canvas"].draw()

        # Update diagnosis distribution
        ax = self.performance_diagnosis_frame["ax"]
        ax.clear()
        if 'diagnosis' in self.data.columns:
            diagnosis_counts = self.data['diagnosis'].value_counts().head(10)
            ax.barh(diagnosis_counts.index, diagnosis_counts.values, color='salmon')
            ax.set_xlabel('Number of Cases', color='white')
            ax.set_ylabel('Category', color='white')
        self.performance_diagnosis_frame["canvas"].draw()

        # Update monthly trend
        ax = self.performance_trend_frame["ax"]
        ax.clear()
        if 'date' in self.data.columns:
            monthly_data = self.data.groupby(self.data['date'].dt.to_period('M')).size()
            ax.plot(range(len(monthly_data)), monthly_data.values, marker='o', color='lightblue')
            ax.set_xlabel('Month', color='white')
            ax.set_ylabel('Number of Cases', color='white')
            ax.set_xticks(range(len(monthly_data)))
            ax.set_xticklabels([str(p) for p in monthly_data.index], rotation=45)
        self.performance_trend_frame["canvas"].draw()

        # Update yearly trend
        ax = self.performance_yearly_trend_frame["ax"]
        ax.clear()
        if 'date' in self.data.columns:
            yearly_data = self.data.groupby(self.data['date'].dt.year).agg({
                'amount': ['count', 'mean']
            })
            
            ax1 = ax
            ax2 = ax1.twinx()
            
            bars = ax1.bar(yearly_data.index, yearly_data[('amount', 'count')], color='lightblue', alpha=0.7)
            line = ax2.plot(yearly_data.index, yearly_data[('amount', 'mean')], color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Year', color='white')
            ax1.set_ylabel('Number of Cases', color='lightblue')
            ax2.set_ylabel('Average Amount', color='lightgreen')
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        self.performance_yearly_trend_frame["canvas"].draw()

        # Update average amount by age
        ax = self.performance_avg_amount_frame["ax"]
        ax.clear()
        if all(col in self.data.columns for col in ['age', 'amount']):
            age_bins = [0, 18, 30, 45, 60, 75, 100]
            age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
            self.data['age_group'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)
            avg_by_age = self.data.groupby('age_group')['amount'].mean()
            
            bars = ax.bar(avg_by_age.index, avg_by_age.values, color='lightblue')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', color='white')
                
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Average Amount ($)', color='white')
            plt.setp(ax.get_xticklabels(), rotation=45)
            
        self.performance_avg_amount_frame["canvas"].draw()

        # Update gender distribution
        ax = self.performance_gender_dist_frame["ax"]
        ax.clear()
        if 'gender' in self.data.columns:
            gender_age_stats = self.data.groupby(['gender', pd.cut(self.data['age'], 
                                                bins=[0, 18, 30, 45, 60, 75, 100],
                                                labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
                                                )])['amount'].agg(['count', 'mean']).reset_index()
            
            bar_width = 0.35
            index = np.arange(len(gender_age_stats['age'].unique()))
            
            male_data = gender_age_stats[gender_age_stats['gender'] == 'M']
            female_data = gender_age_stats[gender_age_stats['gender'] == 'F']
            
            ax.bar(index - bar_width/2, male_data['count'], bar_width, label='Male', color='lightblue')
            ax.bar(index + bar_width/2, female_data['count'], bar_width, label='Female', color='lightpink')
            
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Number of Cases', color='white')
            ax.set_xticks(index)
            ax.set_xticklabels(gender_age_stats['age'].unique(), rotation=45)
            ax.legend()
            
        self.performance_gender_dist_frame["canvas"].draw()

        # Update seasonal pattern
        ax = self.performance_seasonal_frame["ax"]
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
            ax1.set_ylabel('Number of Cases', color='lightblue')
            ax2.set_ylabel('Average Amount', color='lightgreen')
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        self.performance_seasonal_frame["canvas"].draw()

    def update_claim_plots(self):
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
            
            ax1 = ax
            ax2 = ax1.twinx()
            
            bars = ax1.bar(yearly_data.index, yearly_data[('amount', 'count')], color='lightblue', alpha=0.7)
            line = ax2.plot(yearly_data.index, yearly_data[('amount', 'mean')], color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Year', color='white')
            ax1.set_ylabel('Number of Claims', color='lightblue')
            ax2.set_ylabel('Average Claim Amount', color='lightgreen')
            
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
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        self.seasonal_frame["canvas"].draw()

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
        fig = self.correlation_heatmap_frame["fig"]
        
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
        
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
        self.correlation_heatmap_frame["canvas"].draw()

        # Update pairwise relationships
        ax = self.correlation_pairs_frame["ax"]
        ax.clear()
        fig = self.correlation_pairs_frame["fig"]
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)
        
        top_features = abs(corr_matrix['amount']).sort_values(ascending=False)[:4].index
        
        n_features = len(top_features)
        for i in range(n_features):
            for j in range(n_features):
                plt_ax = ax.inset_axes([0.25*i, 0.25*j, 0.23, 0.23])
                if i != j:
                    plt_ax.scatter(self.data[top_features[i]], 
                                 self.data[top_features[j]], 
                                 alpha=0.5, 
                                 c='lightblue',
                                 s=20)
                else:
                    plt_ax.hist(self.data[top_features[i]], bins=20, color='lightgreen')
                if i == 0:
                    plt_ax.set_ylabel(top_features[j], color='white')
                if j == n_features-1:
                    plt_ax.set_xlabel(top_features[i], color='white')
                plt_ax.tick_params(colors='white', labelsize=8)
        
        ax.set_title('Pairwise Relationships of Top Features', color='white', pad=20)
        self.correlation_pairs_frame["canvas"].draw()

        # Update top correlations
        ax = self.top_correlations_frame["ax"]
        ax.clear()
        fig = self.top_correlations_frame["fig"]
        fig.subplots_adjust(left=0.3, right=0.95, bottom=0.2, top=0.9)
        
        top_corr = corr_matrix['amount'].sort_values(ascending=True)
        top_corr = top_corr.drop('amount')
        
        colors = ['red' if x < 0 else 'green' for x in top_corr]
        bars = ax.barh(range(len(top_corr)), top_corr, color=colors)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index, fontsize=8)
        ax.set_xlabel('Correlation Coefficient', color='white')
        ax.set_title('Correlations with Claim Amount', color='white', pad=20)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, i, f'{width:.2f}', 
                    color='white', 
                    va='center',
                    ha='left' if width >= 0 else 'right')
        
        self.top_correlations_frame["canvas"].draw()

        # Update scatter plots
        ax = self.correlation_scatter_frame["ax"]
        ax.clear()
        fig = self.correlation_scatter_frame["fig"]
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
        
        top_vars = abs(corr_matrix['amount']).sort_values(ascending=False)[1:4].index
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_vars)))
        for var, color in zip(top_vars, colors):
            ax.scatter(self.data[var], 
                      self.data['amount'], 
                      alpha=0.5, 
                      c=[color], 
                      label=var)
        
        ax.set_xlabel('Feature Value', color='white')
        ax.set_ylabel('Claim Amount', color='white')
        ax.legend()
        ax.set_title('Top Correlating Variables vs Claims', color='white', pad=20)
        
        self.correlation_scatter_frame["canvas"].draw()

    def upload_file(self):
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
            
            # Update data preview
            self.update_data_preview()
            
            # Update all visualizations
            self.update_performance_plots()
            self.update_claim_plots()
            self.update_correlation_plots()
            
            self.status_label.configure(text=f"Data loaded successfully: {len(self.data)} records")

        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            self.status_label.configure(text=error_msg)
            messagebox.showerror("Error", error_msg)
            print("Detailed error information:")
            print(traceback.format_exc())

    def update_data_preview(self):
        if self.data is None:
            return
            
        # Clear existing preview
        self.preview_text.delete(1.0, tk.END)
        
        # Add basic information
        info_text = f"Dataset Information:\n"
        info_text += f"{'='*50}\n"
        info_text += f"Total Records: {len(self.data):,}\n"
        info_text += f"Total Columns: {len(self.data.columns)}\n"
        info_text += f"Columns: {', '.join(self.data.columns)}\n\n"
        
        # Add basic statistics
        info_text += f"Basic Statistics:\n"
        info_text += f"{'='*50}\n"
        stats = self.data.describe().round(2)
        info_text += stats.to_string()
        info_text += f"\n\n"
        
        # Add first few rows
        info_text += f"First 5 Rows of Data:\n"
        info_text += f"{'='*50}\n"
        info_text += self.data.head().to_string()
        
        # Update preview text
        self.preview_text.insert(1.0, info_text)

if __name__ == "__main__":
    try:
        app = GeneralPricingSystem()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())