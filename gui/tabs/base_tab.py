# gui/tabs/base_tab.py
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils.plot_utils import PlotUtils

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