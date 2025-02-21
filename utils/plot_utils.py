# utils/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns

class PlotUtils:
    @staticmethod
    def setup_dark_style(ax):
        """Setup dark style for matplotlib plots"""
        ax.set_facecolor('#2B2B2B')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_xlabel(ax.get_xlabel(), color='white')
        ax.set_ylabel(ax.get_ylabel(), color='white')
        if ax.get_title():
            ax.set_title(ax.get_title(), color='white', pad=20)

    @staticmethod
    def setup_figure(fig):
        """Setup figure for dark theme"""
        fig.patch.set_facecolor('#2B2B2B')
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)