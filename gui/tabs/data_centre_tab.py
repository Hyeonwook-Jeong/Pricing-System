# gui/tabs/data_centre_tab.py
from gui.tabs.base_tab import BaseTab
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from pathlib import Path
import shutil

class DataCentreTab(BaseTab):
    def setup_ui(self):
        """Setup Data Centre tab UI components"""
        # Create main scrollable frame
        self.main_scrollable = ctk.CTkScrollableFrame(self)
        self.main_scrollable.pack(fill="both", expand=True)
        
        # Main container frame with fixed size for data selection
        self.main_container = ctk.CTkFrame(self.main_scrollable)
        self.main_container.pack(fill="x", padx=20, pady=(20,5))
        
        # Initialize selection tracking variables
        self.selected_database = None  # 'plsql' or 'local'
        self.selected_dataset = None
        
        # Create UI components
        self.create_data_selection_frame()
        self.create_data_filtering_frame()

    def create_section_label(self, parent, text):
        """Create section label"""
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=("Helvetica", 16, "bold")
        )
        label.pack(anchor="w", padx=10, pady=5)

    def create_data_selection_frame(self):
        """Create data selection section"""
        # Main Label
        self.create_section_label(self.main_container, "Data Selection")
        
        # Create PLSQL Database Section
        self.create_plsql_database_section()
        
        # Create Local Database Section
        self.create_local_database_section()
        
        # Create Confirm Button
        self.create_confirm_button()

    def create_plsql_database_section(self):
        """Create PLSQL database section"""
        # PLSQL Database Frame
        self.plsql_frame = ctk.CTkFrame(self.main_container)
        self.plsql_frame.pack(fill="x", padx=10, pady=5)
        
        # Database Label
        self.create_section_label(self.plsql_frame, "PLSQL Database")
        
        # PLSQL Database Listbox
        self.plsql_listbox_frame = ctk.CTkFrame(self.plsql_frame)
        self.plsql_listbox_frame.pack(fill="x", padx=10, pady=5)
        
        self.plsql_listbox = tk.Listbox(
            self.plsql_listbox_frame,
            bg='#2B2B2B',
            fg='white',
            selectmode=tk.SINGLE,
            height=8
        )
        self.plsql_listbox.pack(side="left", fill="x", expand=True)
        self.plsql_listbox.bind('<<ListboxSelect>>', self.on_plsql_select)
        
        self.plsql_scrollbar = tk.Scrollbar(self.plsql_listbox_frame, orient="vertical")
        self.plsql_scrollbar.pack(side="right", fill="y")
        
        self.plsql_listbox.config(yscrollcommand=self.plsql_scrollbar.set)
        self.plsql_scrollbar.config(command=self.plsql_listbox.yview)

        # PLSQL Update Section
        self.plsql_update_frame = ctk.CTkFrame(self.plsql_frame)
        self.plsql_update_frame.pack(fill="x", padx=10, pady=5)
        
        self.plsql_entry = ctk.CTkEntry(
            self.plsql_update_frame,
            placeholder_text="Name of Dataset"
        )
        self.plsql_entry.pack(side="left", fill="x", expand=True, padx=(0,10))
        
        self.plsql_update_button = ctk.CTkButton(
            self.plsql_update_frame,
            text="Update",
            width=100,
            command=self.update_plsql_database
        )
        self.plsql_update_button.pack(side="right")

    def create_local_database_section(self):
        """Create local database section"""
        # Local Database Frame
        self.local_frame = ctk.CTkFrame(self.main_container)
        self.local_frame.pack(fill="x", padx=10, pady=5)
        
        # Local Database Label
        self.create_section_label(self.local_frame, "Local Database")
        
        # Local Database Listbox
        self.local_listbox_frame = ctk.CTkFrame(self.local_frame)
        self.local_listbox_frame.pack(fill="x", padx=10, pady=5)
        
        self.local_listbox = tk.Listbox(
            self.local_listbox_frame,
            bg='#2B2B2B',
            fg='white',
            selectmode=tk.SINGLE,
            height=8
        )
        self.local_listbox.pack(side="left", fill="x", expand=True)
        self.local_listbox.bind('<<ListboxSelect>>', self.on_local_select)
        
        self.local_scrollbar = tk.Scrollbar(self.local_listbox_frame, orient="vertical")
        self.local_scrollbar.pack(side="right", fill="y")
        
        self.local_listbox.config(yscrollcommand=self.local_scrollbar.set)
        self.local_scrollbar.config(command=self.local_listbox.yview)
        
        # File Upload Frame
        self.upload_frame = ctk.CTkFrame(self.local_frame)
        self.upload_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_label = ctk.CTkLabel(
            self.upload_frame,
            text="Please select a file"
        )
        self.file_label.pack(side="left", padx=5, fill="x", expand=True)
        
        self.file_upload_button = ctk.CTkButton(
            self.upload_frame,
            text="Upload Data File",
            command=self.upload_file,
            width=150
        )
        self.file_upload_button.pack(side="right", padx=5)

    def create_confirm_button(self):
        """Create confirm button"""
        self.confirm_frame = ctk.CTkFrame(self.main_container)
        self.confirm_frame.pack(fill="x", padx=10, pady=10)
        
        self.confirm_button = ctk.CTkButton(
            self.confirm_frame,
            text="Confirm",
            width=150,
            command=self.confirm_selection,
            state="disabled"  # Initially disabled
        )
        self.confirm_button.pack(side="right", padx=5)

        # Status label for showing messages
        self.status_label = ctk.CTkLabel(
            self.confirm_frame,
            text="",
            text_color="yellow"
        )
        self.status_label.pack(side="left", padx=5, fill="x", expand=True)

    def create_data_filtering_frame(self):
        """Create data filtering section"""
        # Create filtering frame
        self.filtering_container = ctk.CTkFrame(self.main_scrollable)
        self.filtering_container.pack(fill="x", padx=20, pady=5)
        
        # Filtering Section Label
        self.create_section_label(self.filtering_container, "Data Filtering")
        
        # Create filter options frame
        self.filter_options = ctk.CTkFrame(self.filtering_container)
        self.filter_options.pack(fill="x", padx=10, pady=5)
        
        # Configure grid for filter options
        self.filter_options.grid_columnconfigure(0, weight=1)
        self.filter_options.grid_columnconfigure(1, weight=1)
        
        # Create all filter sections in a 2-column grid
        filters = [
            ("Grouping", "Select your grouping", 
             ["Singapore Individual", "Singapore Corporate", 
              "Dubai Individual", "Summit", "EHP"]),
            ("Country", "Select countries",
             ["United States", "China", "Japan", "Germany", 
              "India", "United Kingdom", "France", "Italy", 
              "Canada", "Brazil", "Russia", "South Korea", 
              "Australia", "Spain", "Mexico", "Indonesia", 
              "Netherlands", "Saudi Arabia", "Switzerland", "Turkey"]),
            ("Continent", "Select continents",
             ["Asia", "Europe", "North America", "South America", 
              "Africa", "Oceania", "Antarctica"]),
            ("Rating Year", "Select rating years",
             [str(year) for year in range(2015, 2025)]),
            ("Start Year", "Select start years",
             [str(year) for year in range(2015, 2025)])
        ]
        
        for i, (label, placeholder, options) in enumerate(filters):
            row = i // 2
            col = i % 2
            self.create_dropdown_filter(label, placeholder, options, row, col)
        
        # Create filter buttons
        self.create_filter_buttons()

    def create_dropdown_filter(self, label_text, placeholder_text, options, row, col):
        """Create a dropdown filter with multi-select capability"""
        # Create frame for this filter
        filter_frame = ctk.CTkFrame(self.filter_options)
        filter_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        # Configure columns
        filter_frame.grid_columnconfigure(1, weight=1)  # Button column
        
        # Create label
        label = ctk.CTkLabel(
            filter_frame,
            text=f"{label_text}:",
            anchor="w"
        )
        label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Create dropdown button that will trigger the selection window
        dropdown_btn = ctk.CTkButton(
            filter_frame,
            text=placeholder_text,
            width=200,
            command=lambda: self.show_selection_window(label_text, options)
        )
        dropdown_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Store the button reference
        setattr(self, f"{label_text.lower().replace(' ', '_')}_btn", dropdown_btn)
        
        # Store selected values
        setattr(self, f"{label_text.lower().replace(' ', '_')}_selected", set())

    def create_filter_buttons(self):
        """Create filter action buttons"""
        # Calculate the row for buttons (after all filters)
        button_row = (len(['Grouping', 'Country', 'Continent', 'Rating Year', 'Start Year']) + 1) // 2
        
        # Apply button
        self.apply_filter_btn = ctk.CTkButton(
            self.filter_options,
            text="Apply Filters",
            command=self.apply_filters,
            width=200
        )
        self.apply_filter_btn.grid(row=button_row, column=0, padx=5, pady=10, sticky="e")
        
        # Reset button
        self.reset_filter_btn = ctk.CTkButton(
            self.filter_options,
            text="Reset Filters",
            command=self.reset_filters,
            width=200
        )
        self.reset_filter_btn.grid(row=button_row, column=1, padx=5, pady=10, sticky="w")

    def show_selection_window(self, title, options):
        """Show a popup window with checkboxes for multiple selection"""
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"Select {title}")
        popup.geometry("600x600")
        
        # Get current selections
        current_selections = getattr(self, f"{title.lower().replace(' ', '_')}_selected")
        
        # Create scrollable frame for options
        scroll_frame = ctk.CTkScrollableFrame(popup)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=(20, 0))
        
        # Dictionary to store checkboxes variables
        checkbox_vars = {}
        
        # Create checkbox for each option
        for option in options:
            var = tk.BooleanVar(value=option in current_selections)
            checkbox_vars[option] = var
            
            checkbox = ctk.CTkCheckBox(
                scroll_frame,
                text=option,
                variable=var,
                width=200
            )
            checkbox.pack(anchor="w", padx=10, pady=2)
        
        # Create buttons frame
        button_frame = ctk.CTkFrame(popup, height=60)
        button_frame.pack(fill="x", padx=20, pady=20, side="bottom")
        button_frame.pack_propagate(False)
        
        # Function to handle selection confirmation
        def confirm_selection():
            selected = {opt for opt, var in checkbox_vars.items() if var.get()}
            attribute_name = f"{title.lower().replace(' ', '_')}_selected"
            setattr(self, attribute_name, selected)
            
            btn = getattr(self, f"{title.lower().replace(' ', '_')}_btn")
            if selected:
                text = f"{len(selected)} item{'s' if len(selected) > 1 else ''} selected"
            else:
                text = f"Select your {title.lower()}"
            btn.configure(text=text)
            
            popup.destroy()
        
        # Function to handle selection cancellation
        def cancel_selection():
            popup.destroy()
        
        # Function to select all options
        def select_all():
            for var in checkbox_vars.values():
                var.set(True)
        
        # Function to clear all selections
        def clear_all():
            for var in checkbox_vars.values():
                var.set(False)
        
        # Create control buttons
        ctk.CTkButton(
            button_frame,
            text="Select All",
            command=select_all,
            width=120,
            height=35
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=clear_all,
            width=120,
            height=35
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=cancel_selection,
            width=120,
            height=35
        ).pack(side="right", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Confirm",
            command=confirm_selection,
            width=120,
            height=35,
            fg_color="#2B6AD0",
            hover_color="#1E4C9A"
        ).pack(side="right", padx=5, pady=10)

    def on_plsql_select(self, event):
        """Handle PLSQL database selection"""
        selection = self.plsql_listbox.curselection()
        if selection:
            self.selected_database = 'plsql'
            self.selected_dataset = self.plsql_listbox.get(selection[0])
            self.local_listbox.selection_clear(0, tk.END)
            self.confirm_button.configure(state="normal")

    def on_local_select(self, event):
        """Handle local database selection"""
        selection = self.local_listbox.curselection()
        if selection:
            self.selected_database = 'local'
            self.selected_dataset = self.local_listbox.get(selection[0])
            self.plsql_listbox.selection_clear(0, tk.END)
            self.confirm_button.configure(state="normal")

    def update_plsql_database(self):
        """Handle PLSQL database update"""
        try:
            dataset_name = self.plsql_entry.get()
            if dataset_name:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.plsql_listbox.insert(tk.END, f"{dataset_name} - Uploaded at {current_time}")
                self.plsql_entry.delete(0, tk.END)
                self.show_status(f"PLSQL database updated with dataset: {dataset_name}")
            else:
                self.show_status("Please enter a dataset name")
        except Exception as e:
            self.show_error("Error updating PLSQL database", e)

    def upload_file(self):
        """Handle file upload"""
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
                return

            file_path = Path(file_path)
            self.file_label.configure(text=str(file_path))
            
            if file_path.suffix.lower() not in ['.csv', '.xlsx', '.xls']:
                raise ValueError("Unsupported file format")

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            file_name = file_path.name
            self.local_listbox.insert(tk.END, f"{file_name} - Uploaded at {current_time}")
            
            # Copy file to current working directory
            target_path = Path.cwd() / file_name
            shutil.copy2(file_path, target_path)
            
            self.show_status(f"File uploaded: {file_name}")
            
        except Exception as e:
            self.show_error("Error uploading file", e)

    def confirm_selection(self):
        """Handle confirm button click"""
        if self.selected_database and self.selected_dataset:
            try:
                dataset_name = self.selected_dataset.split(" - ")[0]
                success, error = self.data_processor.load_file(dataset_name)
                
                if success:
                    self.show_status(f"Successfully loaded dataset: {dataset_name}")
                else:
                    raise Exception(error)
                    
            except Exception as e:
                self.show_error("Error loading dataset", e)
        else:
            self.show_status("Please select a dataset first")

    def apply_filters(self):
        """Apply selected filters to the data"""
        try:
            if not self.data_processor.has_data():
                raise ValueError("No data loaded")
            
            # Get selected items from each filter
            selected_groups = list(self.grouping_selected)
            selected_countries = list(self.country_selected)
            selected_continents = list(self.continent_selected)
            selected_rating_years = list(self.rating_year_selected)
            selected_start_years = list(self.start_year_selected)
            
            # Apply filters
            success, error = self.data_processor.apply_filters(
                groups=selected_groups,
                countries=selected_countries,
                continents=selected_continents,
                rating_years=selected_rating_years,
                start_years=selected_start_years
            )
            
            if success:
                filter_summary = self.data_processor.get_filter_summary()
                self.show_status(f"Filters applied successfully\n{filter_summary}")
            else:
                raise Exception(error)
            
        except Exception as e:
            self.show_error("Error applying filters", e)

    def reset_filters(self):
        """Reset all filters to default values"""
        try:
            # Clear all selections
            filter_types = ['grouping', 'country', 'continent', 'rating_year', 'start_year']
            for filter_type in filter_types:
                setattr(self, f"{filter_type}_selected", set())
                btn = getattr(self, f"{filter_type}_btn")
                btn.configure(text=f"Select your {filter_type}")
            
            # Reset data processor filters
            if self.data_processor.has_data():
                success, error = self.data_processor.reset_filters()
                if success:
                    self.show_status("Filters reset successfully")
                else:
                    raise Exception(error)
                
        except Exception as e:
            self.show_error("Error resetting filters", e)

    def show_error(self, message, error):
        """Display error message"""
        error_msg = f"{message}: {str(error)}"
        self.show_status(error_msg)
        messagebox.showerror("Error", error_msg)

    def show_status(self, message):
        """Display status message"""
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=message)

    def update_view(self):
        """Update view (placeholder for base class compatibility)"""
        pass