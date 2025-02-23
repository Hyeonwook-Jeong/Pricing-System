# gui/tabs/correlation_tab.py
from .base_tab import BaseTab
import customtkinter as ctk
import numpy as np
import seaborn as sns
from utils.plot_utils import PlotUtils
import matplotlib.pyplot as plt
import pandas as pd  # 이 줄을 추가하세요

class CorrelationTab(BaseTab):
    def setup_ui(self):
        """Setup Correlation tab UI components"""
        # Create scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Correlation visualization frames
        self.correlation_frames = ctk.CTkFrame(self.scrollable_frame)
        self.correlation_frames.pack(fill="both", expand=True, padx=5, pady=5)

        # Create graph frames
        self.create_graph_frames()
        self.configure_grid()

    def create_graph_frames(self):
        """Create all graph frames for correlation analysis"""
        self.graphs = {
            'heatmap': self.create_graph_frame(
                self.correlation_frames,
                "Correlation Heatmap",
                row=0, column=0, columnspan=2
            ),
            'pairs': self.create_graph_frame(
                self.correlation_frames,
                "Pairwise Relationships",
                row=1, column=0, columnspan=2
            ),
            'top_correlations': self.create_graph_frame(
                self.correlation_frames,
                "Top 10 Correlations",
                row=2, column=0
            ),
            'scatter': self.create_graph_frame(
                self.correlation_frames,
                "Key Variables vs Claims",
                row=2, column=1
            )
        }

    def configure_grid(self):
        """Configure grid layout"""
        self.correlation_frames.grid_columnconfigure(0, weight=1)
        self.correlation_frames.grid_columnconfigure(1, weight=1)
        for i in range(3):
            self.correlation_frames.grid_rowconfigure(i, weight=1)

    def update_view(self):
        """Update all correlation visualizations"""
        if not self.data_processor.has_data():
            return

        self.update_correlation_heatmap()
        self.update_pairwise_relationships()
        self.update_top_correlations()
        self.update_key_variables_scatter()

    def update_correlation_heatmap(self):
        """Update correlation heatmap"""
        ax = self.graphs['heatmap']['ax']
        ax.clear()
        fig = self.graphs['heatmap']['fig']
        
        corr_matrix = self.data_processor.get_correlation_matrix()
        if corr_matrix is not None:
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
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['heatmap']['canvas'].draw()

    def update_pairwise_relationships(self):
        """Update pairwise relationships visualization"""
        ax = self.graphs['pairs']['ax']
        ax.clear()
        fig = self.graphs['pairs']['fig']
        
        if self.data_processor.has_data():
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)
            
            # Use correlation matrix to get top categorical features
            corr_matrix = self.data_processor.get_correlation_matrix()
            
            # Get top features based on their correlation to each other
            correlation_values = corr_matrix.abs().sum()
            top_features = correlation_values.sort_values(ascending=False)[:4].index
            
            data = self.data_processor.get_data()
            n_features = len(top_features)
            
            for i in range(n_features):
                for j in range(n_features):
                    plt_ax = ax.inset_axes([0.25*i, 0.25*j, 0.23, 0.23])
                    
                    # If not on diagonal, create scatter/bar plot
                    if i != j:
                        # For categorical variables, create stacked bar plot
                        crosstab = pd.crosstab(data[top_features[i]], data[top_features[j]])
                        crosstab.plot(kind='bar', stacked=True, ax=plt_ax, legend=False)
                        plt_ax.set_xlabel('')
                        plt_ax.set_ylabel('')
                    else:
                        # On diagonal, show distribution
                        data[top_features[i]].value_counts().plot(kind='bar', ax=plt_ax)
                        plt_ax.set_title('')

                    if i == 0:
                        plt_ax.set_ylabel(top_features[j], color='white')
                    if j == n_features-1:
                        plt_ax.set_xlabel(top_features[i], color='white')
                    plt_ax.tick_params(colors='white', labelsize=8, rotation=45)
            
            ax.set_title('Pairwise Relationships of Top Categorical Features', color='white', pad=20)
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['pairs']['canvas'].draw()

    def update_top_correlations(self):
        """Update top correlations visualization"""
        ax = self.graphs['top_correlations']['ax']
        ax.clear()
        fig = self.graphs['top_correlations']['fig']
        
        if self.data_processor.has_data():
            fig.subplots_adjust(left=0.3, right=0.95, bottom=0.2, top=0.9)
            
            corr_matrix = self.data_processor.get_correlation_matrix()
            
            # Calculate overall correlation and sort
            correlation_values = corr_matrix.abs().sum()
            top_corr = correlation_values.sort_values(ascending=True)
            top_corr = top_corr.drop(top_corr.index[top_corr.index == top_corr.index[0]])
            
            colors = ['red' if x < 0 else 'green' for x in top_corr]
            bars = ax.barh(range(len(top_corr)), top_corr, color=colors)
            ax.set_yticks(range(len(top_corr)))
            ax.set_yticklabels(top_corr.index, fontsize=8)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('Correlations between Categorical Variables', pad=20)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, i, f'{width:.2f}', 
                    color='white', 
                    va='center',
                    ha='left' if width >= 0 else 'right')
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['top_correlations']['canvas'].draw()

    def update_key_variables_scatter(self):
        """Update key variables scatter plots"""
        ax = self.graphs['scatter']['ax']
        ax.clear()
        fig = self.graphs['scatter']['fig']
        
        if self.data_processor.has_data():
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
            
            corr_matrix = self.data_processor.get_correlation_matrix()
            
            # Get top features based on overall correlation
            correlation_values = corr_matrix.abs().sum()
            top_vars = correlation_values.sort_values(ascending=False)[1:4].index
            
            # For categorical variables, use stacked bar plot
            data = self.data_processor.get_data()
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_vars)))
            for var, color in zip(top_vars, colors):
                # Create a cross-tabulation for the variables
                crosstab = pd.crosstab(data[var], columns='count', normalize='index')
                crosstab.plot(kind='bar', color=color, ax=ax, label=var)
            
            ax.set_xlabel('Categories')
            ax.set_ylabel('Proportion')
            ax.legend()
            ax.set_title('Distribution of Top Categorical Variables', pad=20)
            
        PlotUtils.setup_dark_style(ax)
        self.graphs['scatter']['canvas'].draw()

