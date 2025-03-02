from gui.tabs.base_tab import BaseTab
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from pathlib import Path
import shutil
import os

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
        """Create settings section with username and password (1-1)"""
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
        username_label = ctk.CTkLabel(
            credentials_frame,
            text="Username:"
        )
        username_label.pack(side="left", padx=(10, 5))
        
        self.username_entry = ctk.CTkEntry(
            credentials_frame,
            placeholder_text="Enter your username",
            width=200
        )
        self.username_entry.pack(side="left", padx=5)
        
        # Password (1-1-2)
        password_label = ctk.CTkLabel(
            credentials_frame,
            text="Password:"
        )
        password_label.pack(side="left", padx=(20, 5))
        
        self.password_entry = ctk.CTkEntry(
            credentials_frame,
            placeholder_text="Enter your password",
            width=200,
            show="*"  # Show asterisks for password
        )
        self.password_entry.pack(side="left", padx=5)
        
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
        
        # Listbox frame (1-2-1, 1-3-1)
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
        
        # Store references
        setattr(self, f"{prefix}_listbox", listbox)
        
        # Function to load initial databases
        def load_initial_databases():
            try:
                # 데이터베이스 테이블이 있는 폴더 경로 (수정 필요)
                db_folder = Path('./databases')  # 실제 폴더 경로로 변경
                
                # 폴더가 존재하면
                if db_folder.exists():
                    # CSV와 XLSX 파일 찾기
                    db_files = list(db_folder.glob('*.csv')) + list(db_folder.glob('*.xlsx'))
                    
                    # 리스트박스에 추가
                    for file in db_files:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                        listbox.insert(tk.END, f"{file.stem} - {file.name} - Uploaded at {current_time}")
            
            except Exception as e:
                print(f"Error loading initial databases: {e}")

        # 초기 데이터베이스 로드
        load_initial_databases()
        
        # Entry and button for database upload (1-2-2, 1-2-3, 1-3-2, 1-3-3)
        query_frame = ctk.CTkFrame(panel_frame)
        query_frame.pack(fill="x", padx=10, pady=5)
        
        entry = ctk.CTkEntry(
            query_frame,
            placeholder_text="Enter database name"
        )
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        query_btn = ctk.CTkButton(
            query_frame,
            text="From Database",
            command=lambda p=prefix, e=entry: self.query_database(p, e),
            width=120
        )
        query_btn.pack(side="right")
        
        # Store reference
        setattr(self, f"{prefix}_db_entry", entry)
        
        # Entry and button for file upload (1-2-4, 1-2-5, 1-3-4, 1-3-5)
        upload_frame = ctk.CTkFrame(panel_frame)
        upload_frame.pack(fill="x", padx=10, pady=5)
        
        upload_entry = ctk.CTkEntry(
            upload_frame,
            placeholder_text="Enter file nickname"
        )
        upload_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        upload_btn = ctk.CTkButton(
            upload_frame,
            text="From File",
            command=lambda p=prefix, e=upload_entry: self.upload_file(p, e),
            width=120
        )
        upload_btn.pack(side="right")
        
        # Store reference
        setattr(self, f"{prefix}_file_entry", upload_entry)
        
        # Confirm button (1-2-7, 1-3-7)
        confirm_btn = ctk.CTkButton(
            panel_frame,
            text="Confirm Dataset",
            command=lambda p=prefix: self.confirm_database_selection(p),
            width=200,
            height=30  # 높이를 30으로 줄임
        )
        confirm_btn.pack(anchor="e", padx=10, pady=5, fill="x", expand=True)

        # Status message frame (1-2-6, 1-3-6)
        status_frame = ctk.CTkFrame(panel_frame)
        status_frame.pack(fill="x", padx=10, pady=(0, 10))  # 패딩 조정

        # Status message
        status_label = ctk.CTkLabel(
            status_frame,
            text="",
            text_color="yellow",
            height=20  # 높이를 20으로 줄임
        )
        status_label.pack(fill="x", padx=10, pady=2)
        
        # Store reference
        setattr(self, f"{prefix}_status", status_label)
        
        # Store reference
        setattr(self, f"{prefix}_confirm_btn", confirm_btn)
    
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
    
    # ============== Event Handlers ==============
    
    def save_credentials(self):
        """Handle the credentials confirmation button (1-1-4)"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            self.credentials_status.configure(text="Please enter both username and password")
            return
        
        # Here you would save or validate the credentials
        # For now, we'll just show a success message
        self.credentials_status.configure(text="Credentials saved successfully")
        
        # In a real application, you might store these for database access
        self.db_username = username
        self.db_password = password
    
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
        """Handle database query (1-2-3, 1-3-3)"""
        db_name = entry.get()
        status_label = getattr(self, f"{prefix}_status")
        
        if not db_name:
            status_label.configure(text="Please enter a database name")
            return
            
        try:
            # Simulate database query (you would implement actual query)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            listbox = getattr(self, f"{prefix}_listbox")
            listbox.insert(tk.END, f"{db_name} - Queried at {current_time}")
            
            # Clear entry
            entry.delete(0, tk.END)
            
            status_label.configure(text=f"Successfully queried database: {db_name}")
            
        except Exception as e:
            status_label.configure(text=f"Error querying database: {str(e)}")
    
    def upload_file(self, prefix, entry):
        """Handle file upload (1-2-5, 1-3-5)"""
        file_nickname = entry.get()
        status_label = getattr(self, f"{prefix}_status")
        listbox = getattr(self, f"{prefix}_listbox")
        confirm_btn = getattr(self, f"{prefix}_confirm_btn")
        
        if not file_nickname:
            status_label.configure(text="Please enter a file nickname")
            return
        
        try:
            # Open file selection dialog
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
                return  # User canceled file selection
            
            file_path = Path(file_path)
            
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Add new item to listbox (keeping existing items)
            listbox.insert(tk.END, f"{file_nickname} - {file_path.name} - Uploaded at {current_time}")
            
            # Enable confirm button
            confirm_btn.configure(state="normal")
            
            # Clear input field
            entry.delete(0, tk.END)
            
            # Update status
            status_label.configure(text=f"Successfully uploaded file: {file_nickname}")
            
            # DO NOT call enable_filter_controls here
            
        except Exception as e:
            status_label.configure(text=f"Error uploading file: {str(e)}")

    def confirm_database_selection(self, prefix):
        """Handle database selection confirmation (1-2-7, 1-3-7)"""
        listbox = getattr(self, f"{prefix}_listbox")
        status_label = getattr(self, f"{prefix}_status")
        confirm_btn = getattr(self, f"{prefix}_confirm_btn")
        
        selection = listbox.curselection()
        if not selection:
            status_label.configure(text="Please select a dataset first")
            return
        
        selected_item = listbox.get(selection[0])
        
        # Extract dataset name (remove timestamp and file path)
        parts = selected_item.split(" - ")
        dataset_name = parts[0]
        file_name = parts[1] if len(parts) > 1 else ""
        
        try:
            # Load the file into data processor
            success, error = self.data_processor.load_file(file_name, prefix)
            
            if success:
                # Update status label
                status_label.configure(text=f"Successfully loaded dataset: {dataset_name}")
                
                # Store selected dataset
                self.selected_database = prefix
                self.selected_dataset = dataset_name
                
                # Update filter section based on selected database type
                if prefix == 'standard':
                    # Update filter options without disabling anything
                    self.update_filter_options()
                    self.filter_status.configure(text="")
                else:  # Claim data
                    # Show message without disabling controls
                    self.filter_status.configure(text="Filtering is only available for Standard data")
                    
                # Automatically update the appropriate tab 
                # This ensures visualization updates immediately on data load
                main_app = self.master.master.master
                if prefix == 'standard':
                    for tab_name, tab in main_app.tabs.items():
                        if tab_name == 'Performance':
                            tab.update_view()
                            break
                elif prefix == 'claim':
                    for tab_name, tab in main_app.tabs.items():
                        if tab_name == 'Claim':
                            tab.update_view()
                            break
            else:
                raise Exception(error)
                    
        except Exception as e:
            status_label.configure(text=f"Error loading dataset: {str(e)}")

    def enable_filter_controls(self, enable=True):
        """This method is no longer used to disable controls, only sets status message"""
        # This method now only sets status message without changing button states
        if not enable:
            self.filter_status.configure(text="Filtering is only available for Standard data")
        else:
            self.filter_status.configure(text="")
    
    def open_filter_selection(self, filter_name):
        """Open multi-select window for filters (2-1, 2-2, 2-3, 2-4)"""
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
        
        if not options:
            label = ctk.CTkLabel(scroll_frame, text="No options available")
            label.pack(padx=10, pady=10)
        else:
            # Create checkbox for each option
            for option in options:
                var = tk.BooleanVar(value=option in current_selections)
                checkbox_vars[option] = var
                
                checkbox = ctk.CTkCheckBox(
                    scroll_frame,
                    text=str(option),
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
            setattr(self, attr_name, selected)
            
            btn = getattr(self, f"{filter_name.lower().replace(' ', '_')}_btn")
            if selected:
                text = f"{len(selected)} item{'s' if len(selected) > 1 else ''} selected"
            else:
                text = f"Select {filter_name.lower()}"
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
        # Check if there's any data at all
        if not hasattr(self.data_processor, 'standard_data') or self.data_processor.standard_data is None:
            # No data loaded yet, return empty list instead of error
            return []
        
        try:
            # Handle each filter type directly instead of using get_unique_values
            if filter_name == "Grouping":
                if 'group' in self.data_processor.standard_data.columns:
                    options = sorted(self.data_processor.standard_data['group'].unique())
                    if len(options) == 0:
                        self.filter_status.configure(text="No grouping options available in the data")
                    return options
            elif filter_name == "Country":
                if 'country' in self.data_processor.standard_data.columns:
                    options = sorted(self.data_processor.standard_data['country'].unique())
                    if len(options) == 0:
                        self.filter_status.configure(text="No country options available in the data")
                    return options
            elif filter_name == "Continent":
                if 'continent' in self.data_processor.standard_data.columns:
                    options = sorted(self.data_processor.standard_data['continent'].unique())
                    if len(options) == 0:
                        self.filter_status.configure(text="No continent options available in the data")
                    return options
            elif filter_name == "Rating Year":
                if 'rating_year' in self.data_processor.standard_data.columns:
                    options = sorted([str(year) for year in self.data_processor.standard_data['rating_year'].unique()])
                    if len(options) == 0:
                        self.filter_status.configure(text="No rating year options available in the data")
                    return options
        except Exception as e:
            # Log the specific error
            print(f"Error getting filter options for {filter_name}: {str(e)}")
            self.filter_status.configure(text=f"Error getting options for {filter_name}")
        
        return []
    
    def update_filter_options(self):
        """Update available filter options based on loaded data"""
        # Check if we have standard data
        if not hasattr(self.data_processor, 'standard_data') or self.data_processor.standard_data is None:
            print("No standard data available to update filter options")
            return
        
        try:
            # Update filter options without changing current selections
            for filter_name in ['Grouping', 'Country', 'Continent', 'Rating Year']:
                try:
                    # 필터 옵션 가져오기
                    options = self.get_filter_options(filter_name)
                    
                    if not options:
                        # 옵션이 없을 경우 에러 메시지 업데이트
                        btn = getattr(self, f"{filter_name.lower().replace(' ', '_')}_btn", None)
                        if btn:
                            btn.configure(text=f"Select {filter_name}")
                        continue
                    
                    # 필터 선택값은 초기화하지 않고 옵션만 갱신
                    attr_name = f"{filter_name.lower().replace(' ', '_')}_selected"
                    current_selected = getattr(self, attr_name, set())
                    
                    # 실제 버튼 텍스트 업데이트
                    btn = getattr(self, f"{filter_name.lower().replace(' ', '_')}_btn", None)
                    if btn:
                        if current_selected:
                            btn.configure(text=f"{len(current_selected)} items selected")
                        else:
                            btn.configure(text=f"Select {filter_name}")
                            
                except Exception as e:
                    print(f"Error updating {filter_name} filter: {str(e)}")
            
            # 필터 옵션이 정상적으로 로드되었음을 표시
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
                self.filter_status.configure(text="Filters applied successfully")
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
                
                # 필터가 리셋되었다는 메시지 표시
                self.filter_status.configure(text="Filters reset successfully")
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
            
            # 선택된 데이터베이스 유형에 따라 업데이트
            if self.selected_database == 'standard':
                # 데이터 프로세서의 active_data_type 설정
                self.data_processor.active_data_type = 'standard'
                
                # Performance 탭 업데이트
                for tab_name, tab in main_app.tabs.items():
                    if tab_name == 'Performance':
                        tab.update_view()
                        # 일반적인 업데이트 메시지로 변경
                        self.control_status.configure(text="Tables and graphs updated successfully")
                        break
                
            elif self.selected_database == 'claim':
                # 데이터 프로세서의 active_data_type 설정
                self.data_processor.active_data_type = 'claim'
                
                # Claim 탭 업데이트
                for tab_name, tab in main_app.tabs.items():
                    if tab_name == 'Claim':
                        tab.update_view()
                        # 일반적인 업데이트 메시지로 변경
                        self.control_status.configure(text="Tables and graphs updated successfully")
                        break
            
            # 만약 양쪽 데이터 모두 로드되어 있다면 모두 업데이트
            if self.data_processor.has_data('standard') and self.data_processor.has_data('claim'):
                for tab_name, tab in main_app.tabs.items():
                    if tab_name == 'Performance':
                        self.data_processor.active_data_type = 'standard'
                        tab.update_view()
                    elif tab_name == 'Claim':
                        self.data_processor.active_data_type = 'claim'
                        tab.update_view()
                # 일반적인 업데이트 메시지로 변경
                self.control_status.configure(text="All tables and graphs updated successfully")
                
        except Exception as e:
            self.control_status.configure(text=f"Error updating tables and graphs: {str(e)}")
    
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