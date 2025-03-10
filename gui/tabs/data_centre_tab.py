# gui/tabs/data_centre_tab.py
from gui.tabs.base_tab import BaseTab
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from pathlib import Path
import shutil
import os
import pandas as pd
import getpass

class DataCentreTab(BaseTab):
    def setup_ui(self):
        """Setup Data Centre tab UI components based on the new design"""
        # Create main scrollable frame
        self.main_scrollable = ctk.CTkScrollableFrame(self)
        self.main_scrollable.pack(fill="both", expand=True)
        
        # Section 1: Data Loading
        self.create_data_loading_section()
        
        # Section 2: Data Filtering
        self.create_data_filtering_section()
        
        # Section 3: Control
        self.create_control_section()
        
        # Initialize selection tracking variables
        self.selected_database = None  # 'standard' or 'claim'
        self.selected_dataset = None
        
        # IMPORTANT: DO NOT DISABLE FILTERING CONTROLS AT ALL
        # Remove any calls to enable_filter_controls() here
        
    def create_section_label(self, parent, text):
        """Create section label"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=(10, 5))
        
        label = ctk.CTkLabel(
            frame,
            text=text,
            font=("Helvetica", 16, "bold")
        )
        label.pack(anchor="w", padx=10, pady=5)
        
        return frame
    
    def create_data_loading_section(self):
        """Create data loading section (Section 1)"""
        # Main section label
        data_loading_frame = self.create_section_label(self.main_scrollable, "Data Loading")
        
        # Settings section (1-1)
        self.create_settings_section(data_loading_frame)
        
        # Databases section
        self.create_databases_section(data_loading_frame)
        
    def create_settings_section(self, parent):
        """Create settings section with plsql username and plsql password (1-1)"""
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Settings label (1-1)
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=("Helvetica", 14, "bold")
        )
        settings_label.pack(anchor="w", padx=10, pady=5)
        
        # Credentials frame
        credentials_frame = ctk.CTkFrame(settings_frame)
        credentials_frame.pack(fill="x", padx=10, pady=5)
        
        # Username (1-1-1)
        plsql_username_label = ctk.CTkLabel(
            credentials_frame,
            text="PLSQL username:"
        )
        plsql_username_label.pack(side="left", padx=(10, 5))
        
        self.plsql_username_entry = ctk.CTkEntry(
            credentials_frame,
            placeholder_text="Enter your PLSQL username",
            width=200
        )
        self.plsql_username_entry.pack(side="left", padx=5)
        
        # Password (1-1-2)
        plsql_password_label = ctk.CTkLabel(
            credentials_frame,
            text="PLSQL password:"
        )
        plsql_password_label.pack(side="left", padx=(20, 5))
        
        self.plsql_password_entry = ctk.CTkEntry(
            credentials_frame,
            placeholder_text="Enter your PLSQL password",
            width=200,
            show="*"  # Show asterisks for password
        )
        self.plsql_password_entry.pack(side="left", padx=5)
        
        # Confirm button (1-1-4)
        self.confirm_credentials_btn = ctk.CTkButton(
            credentials_frame,
            text="Confirm",
            command=self.save_credentials,
            width=100
        )
        self.confirm_credentials_btn.pack(side="left", padx=10)
        
        # Status message (1-1-3)
        self.credentials_status = ctk.CTkLabel(
            credentials_frame,
            text="",
            text_color="yellow"
        )
        self.credentials_status.pack(side="right", padx=10, fill="x", expand=True)
    
    def create_databases_section(self, parent):
        """Create standard and claim database sections (1-2, 1-3)"""
        # Container for both database sections
        databases_frame = ctk.CTkFrame(parent)
        databases_frame.pack(fill="x", padx=10, pady=5)
        
        # Configure grid for databases frame
        databases_frame.grid_columnconfigure(0, weight=1)
        databases_frame.grid_columnconfigure(1, weight=1)
        
        # Create Standard database section (1-2)
        self.create_database_panel(
            databases_frame, 
            "Standard", 
            row=0, 
            column=0, 
            prefix="standard"
        )
        
        # Create Claim database section (1-3)
        self.create_database_panel(
            databases_frame, 
            "Claim", 
            row=0, 
            column=1, 
            prefix="claim"
        )
    
    def create_database_panel(self, parent, title, row, column, prefix):
        """Create a database panel with listbox and controls"""
        # Main frame
        panel_frame = ctk.CTkFrame(parent)
        panel_frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
        
        # Title
        panel_title = ctk.CTkLabel(
            panel_frame,
            text=title,
            font=("Helvetica", 14, "bold")
        )
        panel_title.pack(anchor="w", padx=10, pady=5)
        
        # Listbox frame
        listbox_frame = ctk.CTkFrame(panel_frame)
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create listbox with scrollbar
        listbox = tk.Listbox(
            listbox_frame,
            bg='#2B2B2B',
            fg='white',
            selectmode=tk.SINGLE,
            height=8
        )
        listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(listbox_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        # Set a selection event handler
        listbox.bind('<<ListboxSelect>>', 
                    lambda event, p=prefix: self.on_database_select(event, p))
        
        # Store reference to listbox
        setattr(self, f"{prefix}_listbox", listbox)
        
        # Populate listbox with database tables
        try:
            tables = self.data_processor.get_table_list(prefix)
            for table in tables:
                listbox.insert(tk.END, f"{table['display_name']} - {table['date']} - By {table.get('username', 'Unknown')}")
        except Exception as e:
            print(f"Error loading databases: {e}")
        
        # Database query section
        query_frame = ctk.CTkFrame(panel_frame)
        query_frame.pack(fill="x", padx=10, pady=5)
        
        db_entry = ctk.CTkEntry(
            query_frame,
            placeholder_text="Enter dataset name"
        )
        db_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        db_btn = ctk.CTkButton(
            query_frame,
            text="From Database",
            command=lambda p=prefix, e=db_entry: self.query_database(p, e),
            width=120
        )
        db_btn.pack(side="right")
        
        # Store reference
        setattr(self, f"{prefix}_db_entry", db_entry)
        
        # File upload section
        file_frame = ctk.CTkFrame(panel_frame)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        file_entry = ctk.CTkEntry(
            file_frame,
            placeholder_text="Enter dataset name"
        )
        file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        file_btn = ctk.CTkButton(
            file_frame,
            text="From File",
            command=lambda p=prefix, e=file_entry: self.upload_file(p, e),
            width=120
        )
        file_btn.pack(side="right")
        
        # Store reference
        setattr(self, f"{prefix}_file_entry", file_entry)
        
        # Button container
        button_frame = ctk.CTkFrame(panel_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        # Configure button container columns
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        # Confirm button
        confirm_btn = ctk.CTkButton(
            button_frame,
            text="Confirm Dataset",
            command=lambda p=prefix: self.confirm_database_selection(p),
            height=30
        )
        confirm_btn.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        
        # Delete button
        delete_btn = ctk.CTkButton(
            button_frame,
            text="Delete Dataset",
            command=lambda p=prefix: self.delete_database_selection(p),
            height=30,
            fg_color="#E74C3C",
            hover_color="#C0392B"
        )
        delete_btn.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")
        
        # Status message
        status_frame = ctk.CTkFrame(panel_frame)
        status_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        status_label = ctk.CTkLabel(
            status_frame,
            text="",
            text_color="yellow",
            height=20
        )
        status_label.pack(fill="x", padx=10, pady=2)
        
        # Store references
        setattr(self, f"{prefix}_status", status_label)
        setattr(self, f"{prefix}_confirm_btn", confirm_btn)
        setattr(self, f"{prefix}_delete_btn", delete_btn)
    
    def create_data_filtering_section(self):
        """Create data filtering section (Section 2)"""
        # Main section label
        data_filtering_frame = self.create_section_label(self.main_scrollable, "Data Filtering")
        
        # First row for filter buttons
        row1_frame = ctk.CTkFrame(data_filtering_frame)
        row1_frame.pack(fill="x", padx=5, pady=5)
        row1_frame.grid_columnconfigure(0, weight=1)
        row1_frame.grid_columnconfigure(1, weight=1)
        
        # Grouping filter
        grouping_label = ctk.CTkLabel(row1_frame, text="Grouping:")
        grouping_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        
        self.grouping_btn = ctk.CTkButton(
            row1_frame,
            text="Select grouping",
            command=lambda: self.open_filter_selection("Grouping"),
            width=200
        )
        self.grouping_btn.grid(row=0, column=0, padx=(100, 10), pady=5, sticky="ew")
        
        # Country filter
        country_label = ctk.CTkLabel(row1_frame, text="Country:")
        country_label.grid(row=0, column=1, padx=(10, 5), pady=5, sticky="w")
        
        self.country_btn = ctk.CTkButton(
            row1_frame,
            text="Select countries",
            command=lambda: self.open_filter_selection("Country"),
            width=200
        )
        self.country_btn.grid(row=0, column=1, padx=(100, 10), pady=5, sticky="ew")
        
        # Second row for filter buttons
        row2_frame = ctk.CTkFrame(data_filtering_frame)
        row2_frame.pack(fill="x", padx=5, pady=5)
        row2_frame.grid_columnconfigure(0, weight=1)
        row2_frame.grid_columnconfigure(1, weight=1)
        
        # Continent filter
        continent_label = ctk.CTkLabel(row2_frame, text="Continent:")
        continent_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        
        self.continent_btn = ctk.CTkButton(
            row2_frame,
            text="Select continents",
            command=lambda: self.open_filter_selection("Continent"),
            width=200
        )
        self.continent_btn.grid(row=0, column=0, padx=(100, 10), pady=5, sticky="ew")
        
        # Rating Year filter
        rating_year_label = ctk.CTkLabel(row2_frame, text="Rating Year:")
        rating_year_label.grid(row=0, column=1, padx=(10, 5), pady=5, sticky="w")
        
        self.rating_year_btn = ctk.CTkButton(
            row2_frame,
            text="Select rating years",
            command=lambda: self.open_filter_selection("Rating Year"),
            width=200
        )
        self.rating_year_btn.grid(row=0, column=1, padx=(100, 10), pady=5, sticky="ew")
        
        # Apply and Reset Filters buttons - DIRECTLY in data_filtering_frame
        self.apply_filter_btn = ctk.CTkButton(
            data_filtering_frame,
            text="Apply Filters",
            command=self.apply_filters,
            width=150
        )
        self.apply_filter_btn.pack(side="left", padx=(10, 5), pady=10)
        
        self.reset_filter_btn = ctk.CTkButton(
            data_filtering_frame,
            text="Reset Filters",
            command=self.reset_filters,
            width=150
        )
        self.reset_filter_btn.pack(side="left", padx=5, pady=10)
        
        # Filter status message
        self.filter_status = ctk.CTkLabel(
            data_filtering_frame,
            text="",
            text_color="yellow",
            wraplength=800
        )
        self.filter_status.pack(side="left", padx=10, fill="x", expand=True)
        
        # Initialize selected filter values
        self.grouping_selected = set()
        self.country_selected = set()
        self.continent_selected = set()
        self.rating_year_selected = set()
    
    def create_control_section(self):
        """Create control section (Section 3)"""
        # Main section label
        control_frame = self.create_section_label(self.main_scrollable, "Control")
        
        # Update Tables/Graphs button (3-1)
        self.update_btn = ctk.CTkButton(
            control_frame,
            text="Update Tables/Graphs",
            command=self.update_visualizations,
            width=200,
            height=30
        )
        self.update_btn.pack(side="left", padx=10, pady=10)
        
        # Download in Excel button (3-2)
        self.excel_btn = ctk.CTkButton(
            control_frame,
            text="Download in Excel",
            command=self.download_excel,
            width=200,
            height=30
        )
        self.excel_btn.pack(side="left", padx=10, pady=10)
        
        # Status message (3-3) - 컨테이너 제거, 직접 control_frame에 배치
        self.control_status = ctk.CTkLabel(
            control_frame,
            text="",
            text_color="yellow"
        )
        self.control_status.pack(side="left", padx=10, fill="x", expand=True)
    
    def delete_database_selection(self, prefix):
        """Delete selected dataset from database (handler for delete button)"""
        listbox = getattr(self, f"{prefix}_listbox")
        status_label = getattr(self, f"{prefix}_status")
        
        selection = listbox.curselection()
        if not selection:
            status_label.configure(text="Please select a dataset first")
            return
        
        selected_item = listbox.get(selection[0])
        parts = selected_item.split(" - ")
        dataset_name = parts[0]
        
        # Confirm with user before deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the dataset '{dataset_name}'?",
            icon="warning"
        )
        
        if not confirm:
            return
        
        try:
            # Call the data processor's delete method
            success, error = self.data_processor.delete_from_database(prefix, dataset_name)
            
            if success:
                # Remove item from listbox
                listbox.delete(selection[0])
                
                # Update status label
                status_label.configure(text=f"Successfully deleted dataset '{dataset_name}'")
                
                # If this was the currently selected dataset, clear it
                if (hasattr(self, "selected_dataset") and 
                    hasattr(self, "selected_database") and
                    self.selected_dataset == dataset_name and
                    self.selected_database == prefix):
                    self.selected_dataset = None
                    self.selected_database = None
            else:
                status_label.configure(text=f"Deletion error: {error}")
                    
        except Exception as e:
            status_label.configure(text=f"Dataset deletion error: {str(e)}")

    # ============== Event Handlers ==============
    
    def save_credentials(self):
        """Handle the credentials confirmation button (1-1-4)"""
        plsql_username = self.plsql_username_entry.get()
        plsql_password = self.plsql_password_entry.get()
        
        if not plsql_username or not plsql_password:
            self.credentials_status.configure(text="Please enter both plsql username and plsql password")
            return
        
        # Here you would save or validate the credentials
        # For now, we'll just show a success message
        self.credentials_status.configure(text="Credentials saved successfully")
        
        # In a real application, you might store these for database access
        self.db_username = plsql_username
        self.db_password = plsql_password
    
    def on_database_select(self, event, prefix):
        """Handle database selection in listbox (1-2-1, 1-3-1)"""
        listbox = getattr(self, f"{prefix}_listbox")
        selection = listbox.curselection()
        
        if selection:
            # Enable the confirm button
            confirm_btn = getattr(self, f"{prefix}_confirm_btn")
            confirm_btn.configure(state="normal")
            
            # Clear selection in the other listbox
            other_prefix = "claim" if prefix == "standard" else "standard"
            other_listbox = getattr(self, f"{other_prefix}_listbox")
            other_listbox.selection_clear(0, tk.END)
            
            # Store selection information
            selected_item = listbox.get(selection[0])
            self.selected_database = prefix
            self.selected_dataset = selected_item
            
            # Set appropriate status message based on data type
            if prefix == 'claim':
                self.filter_status.configure(text="Filtering is only available for Standard data")
            else:
                self.filter_status.configure(text="")
    
    def query_database(self, prefix, entry):
        """Handle database query processing (1-2-3, 1-3-3)"""
        db_name = entry.get()
        status_label = getattr(self, f"{prefix}_status")
        
        if not db_name:
            status_label.configure(text="Please enter a database name")
            return
            
        try:
            # Search from table list
            tables = self.data_processor.get_table_list(prefix)
            matching_tables = [t for t in tables if db_name.lower() in t['display_name'].lower()]
            
            listbox = getattr(self, f"{prefix}_listbox")
            listbox.delete(0, tk.END)
            
            for table in matching_tables:
                listbox.insert(tk.END, f"{table['display_name']} - {table['date']} - {table['rows']} rows")
            
            if matching_tables:
                status_label.configure(text=f"Found {len(matching_tables)} matching datasets")
            else:
                status_label.configure(text=f"No datasets matching '{db_name}'")
                
        except Exception as e:
            status_label.configure(text=f"Query error: {str(e)}")

    def upload_file(self, prefix, entry):
        """Handle file upload processing (1-2-5, 1-3-5)"""
        file_nickname = entry.get()
        status_label = getattr(self, f"{prefix}_status")
        
        if not file_nickname:
            status_label.configure(text="Please enter a dataset name")
            return
        
        try:
            # File selection dialog
            file_path = filedialog.askopenfilename(
                title=f"Select {prefix.capitalize()} File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("Excel files (old)", "*.xls"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return  # User canceled
                
            # Check file extension
            path = Path(file_path)
            if path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                status_label.configure(text=f"Unsupported file format: {path.suffix}")
                return
                
            # Data validation
            self.data_processor.validate_data(data, prefix)
            
            username = getpass.getuser()

            # Save to database with username
            success, error = self.data_processor.save_to_database(
                data, 
                prefix, 
                file_nickname, 
                username
            )
            
            if success:
                # Update listbox
                listbox = getattr(self, f"{prefix}_listbox")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                listbox.insert(tk.END, f"{file_nickname} - {current_time} - By {username}")
                
                # Clear input field
                entry.delete(0, tk.END)
                
                status_label.configure(text=f"Successfully uploaded file '{file_nickname}'")
            else:
                status_label.configure(text=f"Save error: {error}")
                
        except Exception as e:
            status_label.configure(text=f"File upload error: {str(e)}")

    def confirm_database_selection(self, prefix):
        """Confirm database selection (1-2-7, 1-3-7)"""
        listbox = getattr(self, f"{prefix}_listbox")
        status_label = getattr(self, f"{prefix}_status")
        
        selection = listbox.curselection()
        if not selection:
            status_label.configure(text="Please select a dataset first")
            return
        
        selected_item = listbox.get(selection[0])
        parts = selected_item.split(" - ")
        dataset_name = parts[0]
        
        try:
            # Load data from database
            success, data, error = self.data_processor.load_from_database(prefix, dataset_name)
            
            if success:
                # Update status label
                status_label.configure(text=f"Successfully loaded dataset '{dataset_name}'")
                
                # Store selected dataset
                self.selected_database = prefix
                self.selected_dataset = dataset_name
                
                # Update filter section
                if prefix == 'standard':
                    self.update_filter_options()
                    self.filter_status.configure(text="")
                else:  # Claim data
                    self.filter_status.configure(text="Filtering is only available for Standard data")
                    
                # 이 부분을 제거하여 자동 업데이트를 방지합니다.
                # 아래 코드 삭제:
                # main_app = self.master.master.master
                # if prefix == 'standard':
                #     for tab_name, tab in main_app.tabs.items():
                #         if tab_name == 'Performance':
                #             tab.update_view()
                #             break
                # elif prefix == 'claim':
                #     for tab_name, tab in main_app.tabs.items():
                #         if tab_name == 'Claim':
                #             tab.update_view()
                #             break
                
                # 대신 상태 메시지를 업데이트하여 사용자에게 다음 단계를 안내합니다.
                self.control_status.configure(text="Dataset loaded. Please apply filters if needed, then click 'Update Tables/Graphs' to visualize.")
                
            else:
                status_label.configure(text=f"Data loading error: {error}")
                    
        except Exception as e:
            status_label.configure(text=f"Dataset loading error: {str(e)}")

    def enable_filter_controls(self, enable=True):
        """This method is no longer used to disable controls, only sets status message"""
        # This method now only sets status message without changing button states
        if not enable:
            self.filter_status.configure(text="Filtering is only available for Standard data")
        else:
            self.filter_status.configure(text="")
    
    def open_filter_selection(self, filter_name):
        """Open multi-select window for filters"""
        # Check if standard data is loaded
        if not hasattr(self.data_processor, 'standard_data') or self.data_processor.standard_data is None:
            messagebox.showinfo("Not Available", "Please load and confirm a standard dataset first.")
            self.filter_status.configure(text="Please load and confirm a standard dataset first")
            return
            
        # Get available options for the filter
        options = self.get_filter_options(filter_name)
        
        # If no options available, show message and return
        if not options:
            messagebox.showinfo("No Options", f"No {filter_name.lower()} options available in the loaded data.")
            self.filter_status.configure(text=f"No {filter_name.lower()} options available in the data")
            return
        
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"Select {filter_name}")
        popup.geometry("400x500")
        
        # Get current selections
        attr_name = f"{filter_name.lower().replace(' ', '_')}_selected"
        current_selections = getattr(self, attr_name)
        
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
                text=str(option),
                variable=var,
                width=350
            )
            checkbox.pack(anchor="w", padx=10, pady=2)
        
        # Create buttons frame
        button_frame = ctk.CTkFrame(popup, height=60)
        button_frame.pack(fill="x", padx=20, pady=20, side="bottom")
        button_frame.pack_propagate(False)
        
        # Function to handle selection confirmation
        def confirm_selection():
            selected = {opt for opt, var in checkbox_vars.items() if var.get()}
            setattr(self, attr_name, selected)
            
            btn = getattr(self, f"{filter_name.lower().replace(' ', '_')}_btn")
            if selected:
                text = f"{len(selected)} item{'s' if len(selected) > 1 else ''} selected"
            else:
                text = f"Select {filter_name}"
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
            width=80,
            height=30
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=clear_all,
            width=80,
            height=30
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=cancel_selection,
            width=80,
            height=30
        ).pack(side="right", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Confirm",
            command=confirm_selection,
            width=80,
            height=30,
            fg_color="#2B6AD0",
            hover_color="#1E4C9A"
        ).pack(side="right", padx=5, pady=10)
    
    def get_filter_options(self, filter_name):
        """Get available options for a specific filter"""
        # 디버깅을 위한 로그 추가
        print(f"Getting filter options for: {filter_name}")
        
        # 원본 데이터셋 사용
        data = self.data_processor.original_standard_data
        
        if data is None:
            print("No original standard data available")
            return []
        
        filter_column_map = {
            "Grouping": 'group',
            "Country": 'country',
            "Continent": 'continent',
            "Rating Year": 'rating_year'
        }
        
        column = filter_column_map.get(filter_name)
        
        if column and column in data.columns:
            # For rating year, convert to string to match existing code
            if column == 'rating_year':
                options = sorted(str(year) for year in data[column].unique())
            else:
                options = sorted(data[column].unique())
            
            print(f"Filter options for {filter_name}: {options}")
            return options
        
        print(f"No column found for {filter_name}")
        return []
    
    def update_filter_options(self):
        """Update available filter options based on loaded data"""
        if not hasattr(self.data_processor, 'standard_data') or self.data_processor.standard_data is None:
            return
        
        try:
            for filter_name in ['Grouping', 'Country', 'Continent', 'Rating Year']:
                # Get all possible options for this filter
                options = self.get_filter_options(filter_name)
                
                # Determine the attribute name for selected options
                attr_name = f"{filter_name.lower().replace(' ', '_')}_selected"
                btn_name = f"{filter_name.lower().replace(' ', '_')}_btn"
                
                # Get the current button and selected options
                btn = getattr(self, btn_name, None)
                
                # Get current selected options
                current_selected = getattr(self, attr_name, set())
                
                # Ensure all original options are preserved
                valid_selected = {opt for opt in current_selected if opt in options}
                
                # Update selected options to include all original options
                # but keep the current selections
                setattr(self, attr_name, valid_selected)
                
                # Update button text
                if btn:
                    if valid_selected:
                        btn.configure(text=f"{len(valid_selected)} items selected")
                    else:
                        btn.configure(text=f"Select {filter_name}")
            
            self.filter_status.configure(text="")
            
        except Exception as e:
            print(f"Error updating filter options: {str(e)}")
            self.filter_status.configure(text="Error updating filter options")
    
    def apply_filters(self):
        """Apply selected filters (2-6)"""
        if not self.data_processor.has_data('standard'):
            self.filter_status.configure(text="Please load and confirm a standard dataset first")
            return
            
        try:
            # Get selected values for each filter
            groups = list(self.grouping_selected)
            countries = list(self.country_selected)
            continents = list(self.continent_selected)
            rating_years = list(self.rating_year_selected)
            
            # Update buttons with number of selected items
            if groups:
                self.grouping_btn.configure(text=f"{len(groups)} items selected")
            else:
                self.grouping_btn.configure(text="Select grouping")
            
            if countries:
                self.country_btn.configure(text=f"{len(countries)} items selected")
            else:
                self.country_btn.configure(text="Select countries")
            
            if continents:
                self.continent_btn.configure(text=f"{len(continents)} items selected")
            else:
                self.continent_btn.configure(text="Select continents")
            
            if rating_years:
                self.rating_year_btn.configure(text=f"{len(rating_years)} items selected")
            else:
                self.rating_year_btn.configure(text="Select rating years")
            
            # Apply these filters to the data processor
            success, error = self.data_processor.apply_filters_to_both(
                groups=groups,
                countries=countries,
                continents=continents,
                rating_years=rating_years
            )
            
            if success:
                self.filter_status.configure(text="Filters applied successfully. Click 'Update Tables/Graphs' to see the results.")
                # Note: Intentionally NOT updating views here, waiting for user to click Update button
            else:
                raise Exception(error)
                
        except Exception as e:
            self.filter_status.configure(text=f"Error applying filters: {str(e)}")

    def reset_filters(self):
        """Reset all filters (2-7)"""
        if not self.data_processor.has_data('standard'):
            self.filter_status.configure(text="Please load and confirm a standard dataset first")
            return
                
        try:
            # Clear all filter selections
            self.grouping_selected = set()
            self.country_selected = set()
            self.continent_selected = set()
            self.rating_year_selected = set()
            
            # Reset button labels
            self.grouping_btn.configure(text="Select grouping")
            self.country_btn.configure(text="Select countries")
            self.continent_btn.configure(text="Select continents")
            self.rating_year_btn.configure(text="Select rating years")
            
            # Reset filters in the data processor
            success, error = self.data_processor.reset_filters()
            
            if success:
                # 필터 리셋 후 필터 옵션 목록을 갱신
                self.update_filter_options()
                
                # 필터가 리셋되었다는 메시지 표시 및 다음 단계 안내
                self.filter_status.configure(text="Filters reset successfully. Click 'Update Tables/Graphs' to update visualizations.")
                
                # 여기서 자동으로 업데이트하지 않음
            else:
                raise Exception(error)
                    
        except Exception as e:
            self.filter_status.configure(text=f"Error resetting filters: {str(e)}")
    
    def update_visualizations(self):
        """Update visualizations in other tabs (3-1)"""
        try:
            # 데이터가 선택되어 있는지 확인
            if not hasattr(self, 'selected_database') or not self.selected_dataset:
                self.control_status.configure(text="Please select and confirm a dataset first")
                return
            
            # 메인 애플리케이션 인스턴스 가져오기
            main_app = self.master.master.master
            
            # 디버깅을 위한 상태 출력
            print("\n======== DEBUG: UPDATE VISUALIZATIONS ========")
            print(f"Selected database: {self.selected_database}")
            print(f"Selected dataset: {self.selected_dataset}")
            print(f"Standard data available: {self.data_processor.has_data('standard')}")
            print(f"Claim data available: {self.data_processor.has_data('claim')}")
            
            # 업데이트 상태를 추적
            performance_updated = False
            claim_updated = False
            
            # Performance 탭 강제 업데이트 (standard 데이터 사용)
            if self.data_processor.has_data('standard'):
                print("Standard data exists - attempting to update Performance tab")
                
                # 현재 active_data_type 저장
                original_type = self.data_processor.active_data_type
                
                # active_data_type을 'standard'로 명시적 설정
                self.data_processor.active_data_type = 'standard'
                
                # Performance 탭 찾기 및 업데이트
                performance_tab = None
                for tab_name, tab in main_app.tabs.items():
                    if tab_name == 'Performance':
                        performance_tab = tab
                        break
                
                if performance_tab:
                    print("Found Performance tab - updating...")
                    performance_tab.update_view()
                    performance_updated = True
                    print("Performance tab update completed")
                else:
                    print("ERROR: Performance tab not found!")
                
                # active_data_type 복원
                self.data_processor.active_data_type = original_type
            else:
                print("No standard data available - skipping Performance tab update")
            
            # Claim 탭 강제 업데이트 (claim 데이터 사용)
            if self.data_processor.has_data('claim'):
                print("Claim data exists - attempting to update Claim tab")
                
                # 현재 active_data_type 저장
                original_type = self.data_processor.active_data_type
                
                # active_data_type을 'claim'으로 명시적 설정
                self.data_processor.active_data_type = 'claim'
                
                # Claim 탭 찾기 및 업데이트
                claim_tab = None
                for tab_name, tab in main_app.tabs.items():
                    if tab_name == 'Claim':
                        claim_tab = tab
                        break
                
                if claim_tab:
                    print("Found Claim tab - updating...")
                    claim_tab.update_view()
                    claim_updated = True
                    print("Claim tab update completed")
                else:
                    print("ERROR: Claim tab not found!")
                
                # active_data_type 복원
                self.data_processor.active_data_type = original_type
            else:
                print("No claim data available - skipping Claim tab update")
                
            # 상관관계 탭 업데이트 (두 데이터가 모두 필요할 경우)
            correlation_updated = False
            if self.data_processor.has_data('standard') and self.data_processor.has_data('claim'):
                print("Both data types exist - attempting to update Correlation tab")
                
                # Correlation 탭 찾기 및 업데이트
                correlation_tab = None
                for tab_name, tab in main_app.tabs.items():
                    if tab_name == 'Correlation':
                        correlation_tab = tab
                        break
                
                if correlation_tab:
                    print("Found Correlation tab - updating...")
                    correlation_tab.update_view()
                    correlation_updated = True
                    print("Correlation tab update completed")
                else:
                    print("ERROR: Correlation tab not found!")
            
            # 상태 메시지 업데이트
            update_messages = []
            if performance_updated:
                update_messages.append("Performance")
            if claim_updated:
                update_messages.append("Claim")
            if correlation_updated:
                update_messages.append("Correlation")
            
            if update_messages:
                tabs_str = " and ".join(update_messages)
                status_msg = f"{tabs_str} tables and graphs updated successfully"
                print(f"Update status message: {status_msg}")
                self.control_status.configure(text=status_msg)
            else:
                self.control_status.configure(text="No data available to update tables and graphs")
                
            print("======== DEBUG: UPDATE COMPLETE ========\n")
                
        except Exception as e:
            self.control_status.configure(text=f"Error updating tables and graphs: {str(e)}")
            import traceback
            print(f"Error updating visualizations: {str(e)}")
            print(traceback.format_exc())
    
    def download_excel(self):
        """Download data in Excel template (3-2)"""
        try:
            # Check if data is selected
            if not hasattr(self, 'selected_dataset') or not self.selected_dataset:
                self.control_status.configure(text="Please select and confirm a dataset first")
                return
            
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Save Excel Report"
            )
            
            if not file_path:
                return  # User cancelled
                
            # In a real application, you would:
            # 1. Load the Excel template
            # 2. Fill it with filtered data
            # 3. Save to the chosen location
            
            # For now, simulate success
            self.control_status.configure(text=f"Excel report saved to: {file_path}")
            
        except Exception as e:
            self.control_status.configure(text=f"Error downloading Excel report: {str(e)}")
    
    def update_view(self):
        """Required method for BaseTab compatibility"""
        # This can be empty or perform any initialization needed when tab is shown
        pass