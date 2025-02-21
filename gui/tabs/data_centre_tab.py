# gui/tabs/data_centre_tab.py
from .base_tab import BaseTab
import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from pathlib import Path
import shutil

class DataCentreTab(BaseTab):
    def setup_ui(self):
        """Setup Data Centre tab UI components"""
        # Main container frame
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=(20,5))
        
        # Initialize selection tracking variables
        self.selected_database = None  # 'plsql' or 'local'
        self.selected_dataset = None
        
        # Create UI components
        self.create_data_selection_frame()

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

    def create_section_label(self, parent, text):
        """Create section label"""
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=("Helvetica", 16, "bold")
        )
        label.pack(anchor="w", padx=10, pady=5)

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
            file_path = tk.filedialog.askopenfilename(
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
            
            # 파일을 현재 작업 디렉토리에 복사
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

    def show_error(self, message, error):
        """Display error message"""
        error_msg = f"{message}: {str(error)}"
        self.show_status(error_msg)
        tk.messagebox.showerror("Error", error_msg)

    def show_status(self, message):
        """Display status message"""
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=message)

    def update_view(self):
        """Update view (placeholder for base class compatibility)"""
        pass