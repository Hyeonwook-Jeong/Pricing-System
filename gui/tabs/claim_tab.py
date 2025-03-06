# gui/tabs/claim_tab.py
from .base_tab import BaseTab
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import numpy as np
from utils.plot_utils import PlotUtils
import matplotlib.pyplot as plt
import pandas as pd

class ClaimTab(BaseTab):
    def setup_ui(self):
        """Setup Claim tab UI components with table + graph layout"""
        # Scrollable container
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Create all visualization sections (table + graph pairs)
        self.create_visualization_sections()

    def create_visualization_sections(self):
        """Create all visualization sections with tables and graphs"""
        # Dictionary to store all UI elements
        self.sections = {}
        
        # Create sections for each visualization
        self.create_section("age", "Claims by Age Group")
        self.create_section("amount", "Claim Amount Distribution")
        self.create_section("diagnosis", "Claims by Diagnosis")
        self.create_section("monthly_trend", "Monthly Claim Trend")
        self.create_section("yearly_trend", "Yearly Claim Trend")
        self.create_section("avg_amount", "Average Claim by Age")
        self.create_section("gender", "Claims by Gender")
        self.create_section("seasonal", "Seasonal Pattern")

    def create_section(self, section_id, title):
        """Create a section with table (left) and graph (right)"""
        # Main section container
        section_frame = ctk.CTkFrame(self.scrollable_frame)
        section_frame.pack(fill="x", expand=True, padx=5, pady=10)
        
        # Title for the section
        title_label = ctk.CTkLabel(
            section_frame,
            text=title,
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=5)
        
        # Container for table and graph
        content_frame = ctk.CTkFrame(section_frame)
        content_frame.pack(fill="x", expand=True, padx=5, pady=5)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Create table (left side)
        table_frame = ctk.CTkFrame(content_frame)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create treeview for table
        tree_frame = ctk.CTkFrame(table_frame)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        tree = ttk.Treeview(tree_frame)
        tree.pack(side="left", fill="both", expand=True)
        
        # Add scrollbar to table
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure dark style for treeview
        style = ttk.Style()
        style.configure("Treeview", 
                        background="#2B2B2B",
                        foreground="white",
                        fieldbackground="#2B2B2B")
        style.map('Treeview', 
                 background=[('selected', '#347083')])
        
        # Create graph (right side)
        graph_dict = self.create_graph_frame(content_frame, "", row=0, column=1)
        
        # Store references to UI elements
        self.sections[section_id] = {
            "frame": section_frame,
            "table": tree,
            "graph": graph_dict
        }

    def update_view(self):
        """Update all claim visualizations"""
        # Debug message
        print("\n======== DEBUG: CLAIM TAB UPDATE_VIEW ========")
        print(f"Current active_data_type: {self.data_processor.active_data_type}")
        print(f"Claim data available: {self.data_processor.has_data('claim')}")
        
        if not self.data_processor.has_data('claim'):
            print("No claim data available to update Claim visualizations")
            print("======== DEBUG: CLAIM UPDATE SKIPPED ========\n")
            return

        try:
            # Always use claim data
            original_type = self.data_processor.active_data_type
            self.data_processor.active_data_type = 'claim'
            print(f"Set active_data_type to 'claim' (was {original_type})")
            
            print("Updating claim visualizations...")
            self.update_age_distribution()
            print("- Age distribution updated")
            self.update_amount_distribution()
            print("- Amount distribution updated")
            self.update_diagnosis_distribution()
            print("- Diagnosis distribution updated")
            self.update_monthly_trend()
            print("- Monthly trend updated")
            self.update_yearly_trend()
            print("- Yearly trend updated")
            self.update_average_amount()
            print("- Average amount updated")
            self.update_gender_distribution()
            print("- Gender distribution updated")
            self.update_seasonal_pattern()
            print("- Seasonal pattern updated")
            
            # Restore original data type
            self.data_processor.active_data_type = original_type
            print(f"Restored active_data_type to '{original_type}'")
            print("======== DEBUG: CLAIM UPDATE COMPLETE ========\n")
            
        except Exception as e:
            print(f"Error updating Claim visualizations: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("======== DEBUG: CLAIM UPDATE FAILED ========\n")
            raise e

    # def update_table(self, section_id, data):
    #     """Update table with data"""
    #     if section_id not in self.sections:
    #         return
            
    #     tree = self.sections[section_id]["table"]
        
    #     # Clear existing data
    #     tree.delete(*tree.get_children())
        
    #     # Reset columns
    #     for col in tree["columns"]:
    #         tree.heading(col, text="")
        
    #     if isinstance(data, pd.Series):
    #         # Convert series to dataframe
    #         data = data.reset_index()
    #         columns = data.columns.tolist()
            
    #         # Configure columns
    #         tree["columns"] = columns
    #         for col in columns:
    #             tree.heading(col, text=col)
            
    #         # Add data rows
    #         for i, row in data.iterrows():
    #             values = row.tolist()
    #             tree.insert("", "end", values=values)
                
    #     elif isinstance(data, pd.DataFrame):
    #         columns = data.columns.tolist()
            
    #         # Configure columns
    #         tree["columns"] = columns
    #         for col in columns:
    #             tree.heading(col, text=col)
            
    #         # Add data rows
    #         for i, row in data.iterrows():
    #             values = row.tolist()
    #             tree.insert("", "end", values=values)
                
    #     elif isinstance(data, np.ndarray):
    #         # For numpy arrays, create simple index
    #         tree["columns"] = ["Value"]
    #         tree.heading("Value", text="Value")
            
    #         for i, value in enumerate(data):
    #             tree.insert("", "end", values=[value])

    def update_age_distribution(self):
        """Update age distribution graph and table"""
        section_id = "age"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        age_dist = self.data_processor.get_age_distribution()
        if age_dist is not None:
            # Update table
            self.update_table(section_id, age_dist)
            
            # Update graph
            ax.bar(range(len(age_dist)), age_dist.values, color='skyblue')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Number of Claims')
            ax.set_xticks(range(len(age_dist)))
            ax.set_xticklabels(age_dist.index, rotation=45)
            
        PlotUtils.setup_dark_style(ax)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_amount_distribution(self):
        """Update amount distribution graph and table"""
        section_id = "amount"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        amount_data = self.data_processor.get_amount_distribution()
        if amount_data is not None:
            # Create a histogram and get bin data for the table
            counts, bins = np.histogram(amount_data, bins=50)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            hist_data = pd.DataFrame({
                'Bin_Start': bins[:-1],
                'Bin_End': bins[1:],
                'Count': counts
            })
            
            # Update table
            self.update_table(section_id, hist_data)
            
            # Update graph
            ax.hist(amount_data, bins=50, color='lightgreen')
            ax.set_xlabel('Claim Amount')
            ax.set_ylabel('Frequency')
            
        PlotUtils.setup_dark_style(ax)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_diagnosis_distribution(self):
        """Update diagnosis distribution graph and table"""
        section_id = "diagnosis"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        diagnosis_data = self.data_processor.get_diagnosis_distribution()
        if diagnosis_data is not None:
            # Update table
            self.update_table(section_id, diagnosis_data)
            
            # Update graph
            ax.barh(range(len(diagnosis_data)), diagnosis_data.values, color='salmon')
            ax.set_xlabel('Number of Claims')
            ax.set_ylabel('Diagnosis')
            ax.set_yticks(range(len(diagnosis_data)))
            ax.set_yticklabels(diagnosis_data.index)
            
        PlotUtils.setup_dark_style(ax)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_monthly_trend(self):
        """Update monthly trend graph and table"""
        section_id = "monthly_trend"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        monthly_data = self.data_processor.get_monthly_trend()
        if monthly_data is not None:
            # Update table
            self.update_table(section_id, monthly_data)
            
            # Update graph
            ax.plot(range(len(monthly_data)), monthly_data.values, marker='o')
            ax.set_xlabel('Month')
            ax.set_ylabel('Number of Claims')
            ax.set_xticks(range(len(monthly_data)))
            ax.set_xticklabels([str(p) for p in monthly_data.index], rotation=45)
            
        PlotUtils.setup_dark_style(ax)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_yearly_trend(self):
        """Update yearly trend graph and table"""
        section_id = "yearly_trend"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        yearly_data = self.data_processor.get_yearly_trend()
        if yearly_data is not None:
            # Update table
            self.update_table(section_id, yearly_data)
            
            # Update graph
            ax1 = ax
            ax2 = ax1.twinx()
            
            x = range(len(yearly_data.index))
            bars = ax1.bar(x, yearly_data[('amount', 'count')],
                         color='lightblue', alpha=0.7)
            line = ax2.plot(x, yearly_data[('amount', 'mean')],
                          color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of Claims', color='lightblue')
            ax2.set_ylabel('Average Claim Amount', color='lightgreen')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(yearly_data.index, rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        PlotUtils.setup_dark_style(ax)
        PlotUtils.setup_dark_style(ax2)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_average_amount(self):
        """Update average amount by age graph and table"""
        section_id = "avg_amount"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        avg_by_age = self.data_processor.get_average_amount_by_age()
        if avg_by_age is not None:
            # Update table
            self.update_table(section_id, avg_by_age)
            
            # Update graph
            x = range(len(avg_by_age))
            bars = ax.bar(x, avg_by_age.values, color='lightblue')
            
            ax.set_xticks(x)
            ax.set_xticklabels(avg_by_age.index, rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', color='white')
                
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Average Claim Amount ($)')
            
        PlotUtils.setup_dark_style(ax)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_gender_distribution(self):
        """Update gender distribution graph and table"""
        section_id = "gender"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        gender_data = self.data_processor.get_gender_distribution()
        if gender_data is not None:
            # Update table
            self.update_table(section_id, gender_data)
            
            # Group data by age_group
            age_groups = gender_data['age_group'].unique()
            n_groups = len(age_groups)
            
            bar_width = 0.35
            index = np.arange(n_groups)
            
            # Get data for males and females
            male_data = gender_data[gender_data['gender'] == 'M']['count'].values
            female_data = gender_data[gender_data['gender'] == 'F']['count'].values
            
            # Create bars
            ax.bar(index - bar_width/2, male_data, bar_width, 
                  label='Male', color='lightblue')
            ax.bar(index + bar_width/2, female_data, bar_width,
                  label='Female', color='lightpink')
            
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Number of Claims')
            ax.set_xticks(index)
            ax.set_xticklabels(age_groups, rotation=45)
            ax.legend()
            
        PlotUtils.setup_dark_style(ax)
        self.sections[section_id]["graph"]["canvas"].draw()

    def update_seasonal_pattern(self):
        """Update seasonal pattern graph and table"""
        section_id = "seasonal"
        ax = self.sections[section_id]["graph"]["ax"]
        ax.clear()
        
        seasonal_data = self.data_processor.get_seasonal_pattern()
        if seasonal_data is not None:
            # Update table
            self.update_table(section_id, seasonal_data)
            
            ax1 = ax
            ax2 = ax1.twinx()
            
            x = range(len(seasonal_data.index))
            bars = ax1.bar(x, seasonal_data[('amount', 'count')],
                         color='lightblue', alpha=0.7)
            line = ax2.plot(x, seasonal_data[('amount', 'mean')],
                          color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Season')
            ax1.set_ylabel('Number of Claims', color='lightblue')
            ax2.set_ylabel('Average Claim Amount', color='lightgreen')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(seasonal_data.index, rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        PlotUtils.setup_dark_style(ax)
        PlotUtils.setup_dark_style(ax2)
        self.sections[section_id]["graph"]["canvas"].draw()