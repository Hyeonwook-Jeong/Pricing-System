# gui/tabs/base_tab.py
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils.plot_utils import PlotUtils
import tkinter as tk
from tkinter import ttk
import pandas as pd

class BaseTab(ctk.CTkFrame):
    def __init__(self, parent, data_processor):
        super().__init__(parent)
        self.data_processor = data_processor
        self.graphs = {}
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components - to be implemented by child classes"""
        raise NotImplementedError

    def create_graph_frame(self, parent, title, row, column, columnspan=1):
        """Create a frame for matplotlib graph"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=column, sticky="nsew", padx=5, pady=5, columnspan=columnspan)

        # Title
        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=5)

        # matplotlib graph
        fig = Figure(figsize=(8, 6), dpi=100)
        PlotUtils.setup_figure(fig)
        
        ax = fig.add_subplot(111)
        PlotUtils.setup_dark_style(ax)
            
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        return {"frame": frame, "fig": fig, "ax": ax, "canvas": canvas}

    def update_view(self):
        """Update all visualizations - to be implemented by child classes"""
        raise NotImplementedError
    
    def update_table(self, section_id, data):
        """Update table with data"""
        if section_id not in self.sections:
            return
            
        tree = self.sections[section_id]["table"]
        
        # Clear existing data
        tree.delete(*tree.get_children())
        
        # Configure style for more readable view
        style = ttk.Style()
        style.configure("Compact.Treeview", 
                        rowheight=25,  # Slightly increased, but not too much
                        font=('Arial', 10))  # Slightly larger, but not bold
        style.configure("Compact.Treeview.Heading", 
                        font=('Arial', 10))  # Consistent heading font
        tree.configure(style="Compact.Treeview")
        
        # Reset columns
        tree["columns"] = ()
        tree.heading("#0", text="")
        tree.column("#0", width=0, stretch=tk.NO)
        
        if isinstance(data, pd.Series):
            # Convert series to dataframe
            data = data.reset_index()
            columns = data.columns.tolist()
        elif isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
        else:
            columns = ["Value"]
            data = pd.DataFrame({columns[0]: data})
        
        # Configure columns dynamically
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            # More conservative width calculation
            col_width = max(
                len(col) * 10,  # Moderate header width
                max(len(str(val)) * 10 for val in data[col]) if not data[col].empty else 100,
                100  # Minimum width of 100 pixels
            )
            tree.column(col, width=col_width, anchor='center')
        
        # Add data rows
        for i, row in data.iterrows():
            # Convert all values to strings to avoid potential formatting issues
            values = [str(val) for val in row.tolist()]
            tree.insert("", "end", values=values)
        
        # Optional: Add scrollbar if many rows
        for child in tree.master.winfo_children():
            if isinstance(child, ttk.Scrollbar):
                child.destroy()
        
        scrollbar = ttk.Scrollbar(tree.master, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")