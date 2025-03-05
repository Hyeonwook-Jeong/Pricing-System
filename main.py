# main.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import pandas as pd
import traceback
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk

from gui.tabs import (
    DataCentreTab,
    PerformanceTab,
    ClaimTab,
    CorrelationTab
)
from utils.data_processor import DataProcessor

class GeneralPricingSystem(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.data_processor = DataProcessor()
        self.setup_window()
        self.create_gui()
        
        self.current_tab = None

    def setup_window(self):
        """Initialize window settings and theme"""
        self.title("GPS - General Pricing System")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

    def create_gui(self):
        """Create the main GUI components"""
        self.create_main_container()
        self.create_header()
        self.create_status_label()
        self.create_notebook()
        self.setup_tabs()

    def create_main_container(self):
        """Create the main container frame"""
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

    def create_header(self):
        """Create the header with logo and program name"""
        # Header panel
        self.header_panel = ctk.CTkFrame(self.main_container, height=60, fg_color="white")
        self.header_panel.pack(fill="x", padx=5, pady=(5, 0))
        self.header_panel.pack_propagate(False)

        # Logo frame
        self.logo_frame = ctk.CTkFrame(self.header_panel, height=60, fg_color="white")
        self.logo_frame.pack(fill="x", padx=5, pady=2)
        self.logo_frame.pack_propagate(False)

        # Header content
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

        # Allianz logo
        self.load_company_logo()

    def load_company_logo(self):
        """Load and display company logo"""
        try:
            logo_image = Image.open("./utils/_logo_.png")
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
        except Exception as e:
            print(f"Error loading logo: {e}")

    def create_status_label(self):
        """Create status message display"""
        self.status_label = ctk.CTkLabel(
            self.main_container,
            text="",
            text_color="yellow"
        )
        self.status_label.pack(fill="x", padx=5)

    def create_notebook(self):
        """Create notebook for tabs"""
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)

    def setup_tabs(self):
        """Setup all tabs"""
        self.tabs = {
            'Data Centre': DataCentreTab(self.notebook, self.data_processor),
            'Performance': PerformanceTab(self.notebook, self.data_processor),
            'Claim': ClaimTab(self.notebook, self.data_processor),
            'Correlation': CorrelationTab(self.notebook, self.data_processor)
        }

        for name, tab in self.tabs.items():
            self.notebook.add(tab, text=name)

    def on_tab_changed(self, event):
        """Handle tab change events"""
        try:
            current_tab = self.notebook.select()
            tab_name = self.notebook.tab(current_tab, "text")
            self.current_tab = tab_name
            
            if self.data_processor.has_data():
                print(f"Tab changed to: {tab_name}")

        except Exception as e:
            self.show_error("Error updating tab visualizations", e)

    def show_error(self, message, error):
        """Display error message"""
        error_msg = f"{message}: {str(error)}"
        self.status_label.configure(text=error_msg)
        messagebox.showerror("Error", error_msg)
        print(f"Error: {error_msg}")
        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        app = GeneralPricingSystem()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())