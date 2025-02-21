# gui/tabs/performance_tab.py
from .base_tab import BaseTab
import customtkinter as ctk
import numpy as np
from utils.plot_utils import PlotUtils
import matplotlib.pyplot as plt

class PerformanceTab(BaseTab):
    def setup_ui(self):
        """Setup Performance tab UI components"""
        # Scrollable container
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Graph container
        self.graphs_container = ctk.CTkFrame(self.scrollable_frame)
        self.graphs_container.pack(fill="both", expand=True, padx=1, pady=1)

        # Create all graph frames
        self.create_graph_frames()
        self.configure_grid()

    def create_graph_frames(self):
        """Create all graph frames for performance visualization"""
        self.graphs = {
            'age': self.create_graph_frame(
                self.graphs_container,
                "Performance by Age Group",
                row=0, column=0
            ),
            'amount': self.create_graph_frame(
                self.graphs_container,
                "Performance Amount Distribution",
                row=0, column=1
            ),
            'diagnosis': self.create_graph_frame(
                self.graphs_container,
                "Performance by Category",
                row=1, column=0
            ),
            'monthly_trend': self.create_graph_frame(
                self.graphs_container,
                "Monthly Performance Trend",
                row=1, column=1
            ),
            'yearly_trend': self.create_graph_frame(
                self.graphs_container,
                "Yearly Performance Trend",
                row=2, column=0
            ),
            'avg_amount': self.create_graph_frame(
                self.graphs_container,
                "Average Performance by Age",
                row=2, column=1
            ),
            'gender': self.create_graph_frame(
                self.graphs_container,
                "Performance by Gender",
                row=3, column=0
            ),
            'seasonal': self.create_graph_frame(
                self.graphs_container,
                "Seasonal Performance Pattern",
                row=3, column=1
            )
        }

    def configure_grid(self):
        """Configure grid layout"""
        for i in range(4):
            self.graphs_container.grid_columnconfigure(i, weight=1)
            self.graphs_container.grid_rowconfigure(i, weight=1)

    def update_view(self):
        """Update all performance visualizations"""
        if not self.data_processor.has_data():
            return

        try:
            self.update_age_distribution()
            self.update_amount_distribution()
            self.update_diagnosis_distribution()
            self.update_monthly_trend()
            self.update_yearly_trend()
            self.update_average_amount()
            self.update_gender_distribution()
            self.update_seasonal_pattern()
        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
            raise e

    def update_age_distribution(self):
        """Update age distribution graph"""
        ax = self.graphs['age']['ax']
        ax.clear()
        
        age_dist = self.data_processor.get_age_distribution()
        if age_dist is not None:
            ax.bar(range(len(age_dist)), age_dist.values, color='skyblue')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Performance Count')
            ax.set_xticks(range(len(age_dist)))
            ax.set_xticklabels(age_dist.index, rotation=45)
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['age']['canvas'].draw()

    def update_amount_distribution(self):
        """Update amount distribution graph"""
        ax = self.graphs['amount']['ax']
        ax.clear()
        
        amount_data = self.data_processor.get_amount_distribution()
        if amount_data is not None:
            ax.hist(amount_data, bins=50, color='lightgreen')
            ax.set_xlabel('Performance Amount')
            ax.set_ylabel('Frequency')
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['amount']['canvas'].draw()

    def update_diagnosis_distribution(self):
        """Update diagnosis distribution graph"""
        ax = self.graphs['diagnosis']['ax']
        ax.clear()
        
        diagnosis_data = self.data_processor.get_diagnosis_distribution()
        if diagnosis_data is not None:
            ax.barh(range(len(diagnosis_data)), diagnosis_data.values, color='salmon')
            ax.set_xlabel('Performance Count')
            ax.set_ylabel('Category')
            ax.set_yticks(range(len(diagnosis_data)))
            ax.set_yticklabels(diagnosis_data.index)
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['diagnosis']['canvas'].draw()

    def update_monthly_trend(self):
        """Update monthly trend graph"""
        ax = self.graphs['monthly_trend']['ax']
        ax.clear()
        
        monthly_data = self.data_processor.get_monthly_trend()
        if monthly_data is not None:
            ax.plot(range(len(monthly_data)), monthly_data.values, marker='o')
            ax.set_xlabel('Month')
            ax.set_ylabel('Performance Count')
            ax.set_xticks(range(len(monthly_data)))
            ax.set_xticklabels([str(p) for p in monthly_data.index], rotation=45)
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['monthly_trend']['canvas'].draw()

    def update_yearly_trend(self):
        """Update yearly trend graph"""
        ax = self.graphs['yearly_trend']['ax']
        ax.clear()
        
        yearly_data = self.data_processor.get_yearly_trend()
        if yearly_data is not None:
            ax1 = ax
            ax2 = ax1.twinx()
            
            x = range(len(yearly_data.index))
            bars = ax1.bar(x, yearly_data[('amount', 'count')],
                         color='lightblue', alpha=0.7)
            line = ax2.plot(x, yearly_data[('amount', 'mean')],
                          color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of Cases', color='lightblue')
            ax2.set_ylabel('Average Performance Amount', color='lightgreen')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(yearly_data.index, rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        PlotUtils.setup_dark_style(ax)
        PlotUtils.setup_dark_style(ax2)
        self.graphs['yearly_trend']['canvas'].draw()

    def update_average_amount(self):
        """Update average amount by age graph"""
        ax = self.graphs['avg_amount']['ax']
        ax.clear()
        
        avg_by_age = self.data_processor.get_average_amount_by_age()
        if avg_by_age is not None:
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
            ax.set_ylabel('Average Performance Amount ($)')
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['avg_amount']['canvas'].draw()

    def update_gender_distribution(self):
        """Update gender distribution graph"""
        ax = self.graphs['gender']['ax']
        ax.clear()
        
        gender_data = self.data_processor.get_gender_distribution()
        if gender_data is not None:
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
            ax.set_ylabel('Performance Count')
            ax.set_xticks(index)
            ax.set_xticklabels(age_groups, rotation=45)
            ax.legend()
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['gender']['canvas'].draw()

    def update_seasonal_pattern(self):
        """Update seasonal pattern graph"""
        ax = self.graphs['seasonal']['ax']
        ax.clear()
        
        seasonal_data = self.data_processor.get_seasonal_pattern()
        if seasonal_data is not None:
            ax1 = ax
            ax2 = ax1.twinx()
            
            x = range(len(seasonal_data.index))
            bars = ax1.bar(x, seasonal_data[('amount', 'count')],
                         color='lightblue', alpha=0.7)
            line = ax2.plot(x, seasonal_data[('amount', 'mean')],
                          color='lightgreen', marker='o', linewidth=2)
            
            ax1.set_xlabel('Season')
            ax1.set_ylabel('Number of Cases', color='lightblue')
            ax2.set_ylabel('Average Amount', color='lightgreen')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(seasonal_data.index, rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', color='white')
                
        PlotUtils.setup_dark_style(ax)
        PlotUtils.setup_dark_style(ax2)
        self.graphs['seasonal']['canvas'].draw()