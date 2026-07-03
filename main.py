import cv2
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from camera import CameraStream, scan_available_webcams
from hands import AirDrawer, CameraFusionManager

# Design System Colors
BG_COLOR = "#0f172a"          # slate-900 (Main background)
CARD_BG = "#1e293b"           # slate-800 (Card background)
BORDER_COLOR = "#334155"      # slate-700 (Subtle border outline)
TEXT_PRIMARY = "#f8fafc"      # slate-50 (Primary text)
TEXT_SECONDARY = "#94a3b8"    # slate-400 (Secondary/subtle text)
ACCENT_PRIMARY = "#4f46e5"    # indigo-600 (Primary action button)
ACCENT_PRIMARY_HOVER = "#4338ca" # indigo-700 (Hover state)
ACCENT_SECONDARY = "#4b5563"  # gray-600 (Neutral action button)
ACCENT_SECONDARY_HOVER = "#374151" # gray-700 (Neutral hover)
COLOR_SUCCESS = "#10b981"     # emerald-500 (Connected status)
COLOR_DANGER = "#ef4444"      # red-500 (Disconnected status)
COLOR_WARNING = "#f59e0b"     # amber-500 (Connecting status)

# Air Drawing Colors (mapped from GUI name to BGR tuples for OpenCV rendering)
COLOR_MAP = {
    "Indigo": (229, 70, 79),    # BGR for Indigo (#4f46e5)
    "Emerald": (129, 185, 16),  # BGR for Emerald (#10b981)
    "Crimson": (68, 68, 239),   # BGR for Crimson (#ef4444)
    "Amber": (11, 158, 245),    # BGR for Amber (#f59e0b)
    "Slate": (148, 115, 100)    # BGR for Slate Gray (#64748b)
}

class ImagePlaceholderManager:
    """
    Generates and caches customized placeholder frames to show when cameras are offline.
    Prevents recreating images on every frame, which optimizes CPU and memory performance.
    """
    def __init__(self, width=480, height=360):
        self.width = width
        self.height = height
        self.cache = {}

    def get_placeholder(self, status, camera_name):
        """
        Retrieves the styled placeholder image based on status and camera name.
        Uses cached image if already generated.
        """
        cache_key = f"{camera_name}_{status}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Draw a custom dark-mode placeholder with camera icon graphics
        img = Image.new("RGB", (self.width, self.height), color=(30, 41, 59)) # slate-800
        draw = ImageDraw.Draw(img)
        
        # Center points
        cx, cy = self.width // 2, self.height // 2
        
        # Color coding for different connection states
        if "Connected" in status:
            status_color = (16, 185, 129)  # Emerald Green
        elif "Connecting" in status:
            status_color = (245, 158, 11)  # Amber Yellow
        elif "Disconnected" in status or "Retrying" in status:
            status_color = (239, 68, 68)   # Crimson Red
        else:
            status_color = (148, 163, 184) # Slate Gray
            
        # Draw camera icon schematic
        # Camera body
        draw.rounded_rectangle([cx - 45, cy - 35, cx + 45, cy + 25], radius=8, fill=(15, 23, 42), outline=(71, 85, 105), width=2)
        # Camera lens flash/top notch
        draw.polygon([(cx - 20, cy - 35), (cx - 12, cy - 45), (cx + 12, cy - 45), (cx + 20, cy - 35)], fill=(15, 23, 42), outline=(71, 85, 105))
        # Inner lens circle
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(30, 41, 59), outline=status_color, width=3)
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=status_color)
        
        # Load standard system fonts
        try:
            font_title = ImageFont.truetype("segoeui.ttf", 18)
            font_status = ImageFont.truetype("segoeui.ttf", 13)
        except IOError:
            font_title = ImageFont.load_default()
            font_status = ImageFont.load_default()

        # Draw labels centered
        def draw_centered_text(text, y_pos, font, color):
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
            else:
                text_width = draw.textsize(text, font=font)[0]
            draw.text((cx - text_width // 2, y_pos), text, fill=color, font=font)

        draw_centered_text(camera_name, cy + 45, font_title, (241, 245, 249))
        draw_centered_text(status, cy + 72, font_status, status_color)

        tk_img = ImageTk.PhotoImage(img)
        self.cache[cache_key] = tk_img
        return tk_img


class StyledWidgetsHelper:
    """
    Helper class containing static methods to create uniform, styled Tkinter widgets
    matching our slate dark theme.
    """
    @staticmethod
    def create_card(parent):
        """Creates a container frame styled as a card with a subtle border."""
        return tk.Frame(
            parent,
            bg=CARD_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )

    @staticmethod
    def create_label(parent, text, font_size=10, is_bold=False, fg=TEXT_PRIMARY, bg=CARD_BG):
        """Creates a standard label with custom typography."""
        font_weight = "bold" if is_bold else "normal"
        return tk.Label(
            parent,
            text=text,
            font=("Segoe UI", font_size, font_weight),
            fg=fg,
            bg=bg
        )

    @staticmethod
    def create_button(parent, text, command, bg=ACCENT_PRIMARY, fg="#ffffff", hover_bg=ACCENT_PRIMARY_HOVER):
        """Creates a flat button with custom hover micro-animations."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 9, "bold"),
            bg=bg,
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=12,
            pady=5,
            cursor="hand2"
        )
        
        # Hover effect events
        def on_enter(e):
            btn['bg'] = hover_bg
        def on_leave(e):
            btn['bg'] = bg
            
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    @staticmethod
    def create_entry(parent, width=25):
        """Creates a modern flat entry field with a border highlight on focus."""
        entry = tk.Entry(
            parent,
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_PRIMARY,
            width=width
        )
        return entry


class CameraPortalApp:
    """
    Main desktop application class that coordinates the GUI, local webcam scanning,
    camera threads management, and rendering streams side-by-side.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Camera Sync Portal")
        self.root.geometry("1150x740")
        self.root.configure(bg=BG_COLOR)
        
        # Prevent resizing to dimensions too small to fit feeds
        self.root.minsize(1100, 710)

        # Initialize placeholders
        self.placeholder_manager = ImagePlaceholderManager(480, 360)

        # Initialize camera streams (start as uninitialized/stopped)
        self.webcam_stream = None
        self.android_stream = None
        self.selected_webcam_idx = 0
        
        # Instantiate Multi-Camera Fusion components
        self.shared_drawer = AirDrawer()
        self.fusion_manager = CameraFusionManager(self.shared_drawer)

        # Create structured layout
        self.build_ui()

        # Scan for webcams in a background thread to prevent UI freezing on launch
        self.scan_webcams_async()

        # Handle window closure cleanly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the GUI update loop (15ms interval for ~60fps checking rate)
        self.update_loop()

    def build_ui(self):
        """
        Builds the dark theme user interface containing a header,
        side-by-side feed cards, and footer controller settings.
        """
        # --- TITLE HEADER BAR ---
        header_frame = tk.Frame(self.root, bg=CARD_BG, height=60, bd=0, highlightthickness=1, highlightbackground=BORDER_COLOR)
        header_frame.pack(fill=tk.X, side=tk.TOP, ipady=10)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="DUAL CAMERA PORTAL",
            font=("Segoe UI", 16, "bold"),
            fg=TEXT_PRIMARY,
            bg=CARD_BG
        )
        title_label.pack(side=tk.LEFT, padx=25, pady=5)

        subtitle_label = tk.Label(
            header_frame,
            text="Webcam & Android Camera Sync System (with MediaPipe Hands)",
            font=("Segoe UI", 10),
            fg=TEXT_SECONDARY,
            bg=CARD_BG
        )
        subtitle_label.pack(side=tk.LEFT, padx=5, pady=10)

        # --- MAIN FEED CONTAINERS (SIDE-BY-SIDE) ---
        main_container = tk.Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)

        # 1. Local Webcam Feed Card
        self.webcam_card = StyledWidgetsHelper.create_card(main_container)
        self.webcam_card.grid(row=0, column=0, padx=10, sticky="nsew")

        # Webcam Header
        webcam_header = tk.Frame(self.webcam_card, bg=CARD_BG)
        webcam_header.pack(fill=tk.X, padx=15, pady=10)
        
        self.webcam_title = StyledWidgetsHelper.create_label(webcam_header, "LOCAL WEBCAM", font_size=12, is_bold=True)
        self.webcam_title.pack(side=tk.LEFT)

        self.webcam_status_lbl = StyledWidgetsHelper.create_label(webcam_header, "OFFLINE", font_size=9, fg=COLOR_DANGER)
        self.webcam_status_lbl.pack(side=tk.RIGHT, padx=10)

        self.webcam_fps_lbl = StyledWidgetsHelper.create_label(webcam_header, "FPS: 0.0", font_size=9, fg=TEXT_SECONDARY)
        self.webcam_fps_lbl.pack(side=tk.RIGHT)

        # Webcam Display Label
        self.webcam_display = tk.Label(self.webcam_card, bg=BG_COLOR, bd=0)
        self.webcam_display.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # 2. Android Camera Feed Card
        self.android_card = StyledWidgetsHelper.create_card(main_container)
        self.android_card.grid(row=0, column=1, padx=10, sticky="nsew")

        # Android Header
        android_header = tk.Frame(self.android_card, bg=CARD_BG)
        android_header.pack(fill=tk.X, padx=15, pady=10)

        self.android_title = StyledWidgetsHelper.create_label(android_header, "ANDROID PHONE FEED", font_size=12, is_bold=True)
        self.android_title.pack(side=tk.LEFT)

        self.android_status_lbl = StyledWidgetsHelper.create_label(android_header, "OFFLINE", font_size=9, fg=COLOR_DANGER)
        self.android_status_lbl.pack(side=tk.RIGHT, padx=10)

        self.android_fps_lbl = StyledWidgetsHelper.create_label(android_header, "FPS: 0.0", font_size=9, fg=TEXT_SECONDARY)
        self.android_fps_lbl.pack(side=tk.RIGHT)

        # Android Display Label
        self.android_display = tk.Label(self.android_card, bg=BG_COLOR, bd=0)
        self.android_display.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # --- CONTROL DASHBOARD PANEL ---
        controls_container = StyledWidgetsHelper.create_card(self.root)
        controls_container.pack(fill=tk.X, side=tk.BOTTOM, padx=30, pady=(0, 30), ipady=15)

        controls_container.grid_columnconfigure(0, weight=1)
        controls_container.grid_columnconfigure(1, weight=1)

        # A. Webcam Settings Section (Left Side of Controls)
        webcam_ctrl = tk.Frame(controls_container, bg=CARD_BG)
        webcam_ctrl.grid(row=0, column=0, padx=25, sticky="nsew")

        StyledWidgetsHelper.create_label(webcam_ctrl, "Local Webcam Configuration", font_size=11, is_bold=True).pack(anchor=tk.W, pady=(0, 8))

        webcam_select_row = tk.Frame(webcam_ctrl, bg=CARD_BG)
        webcam_select_row.pack(fill=tk.X, pady=2)

        StyledWidgetsHelper.create_label(webcam_select_row, "Select Device: ", font_size=9, fg=TEXT_SECONDARY).pack(side=tk.LEFT)

        # Styled Combobox (Uses Tkinter ttk styling)
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("TCombobox", fieldbackground=BG_COLOR, background=CARD_BG, foreground="#ffffff", bordercolor=BORDER_COLOR)
        
        self.webcam_dropdown = ttk.Combobox(webcam_select_row, state="readonly", width=22)
        self.webcam_dropdown.pack(side=tk.LEFT, padx=10)
        self.webcam_dropdown['values'] = ["Scanning..."]
        self.webcam_dropdown.current(0)
        self.webcam_dropdown.bind("<<ComboboxSelected>>", self.on_webcam_dropdown_changed)

        self.btn_scan = StyledWidgetsHelper.create_button(
            webcam_select_row, 
            "Rescan Devices", 
            self.scan_webcams_async, 
            bg=ACCENT_SECONDARY, 
            hover_bg=ACCENT_SECONDARY_HOVER
        )
        self.btn_scan.pack(side=tk.LEFT, padx=5)

        # Hand Tracking Checkbox for Webcam
        self.webcam_hand_tracking_var = tk.BooleanVar(value=True)
        self.chk_webcam_tracking = tk.Checkbutton(
            webcam_ctrl,
            text="Enable Hand Tracking (MediaPipe)",
            variable=self.webcam_hand_tracking_var,
            command=self.on_webcam_tracking_toggle,
            font=("Segoe UI", 9),
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            activebackground=CARD_BG,
            activeforeground=TEXT_PRIMARY,
            selectcolor=BG_COLOR,
            cursor="hand2"
        )
        self.chk_webcam_tracking.pack(anchor=tk.W, pady=(6, 4))

        # Start / Stop Toggle Button for Webcam
        self.btn_toggle_webcam = StyledWidgetsHelper.create_button(
            webcam_ctrl,
            "Start Webcam Feed",
            self.toggle_webcam,
            bg=COLOR_SUCCESS,
            hover_bg="#059669" # darker green
        )
        self.btn_toggle_webcam.pack(anchor=tk.W, pady=(5, 0))

        # Drawing Controls Row for Webcam
        webcam_draw_row = tk.Frame(webcam_ctrl, bg=CARD_BG)
        webcam_draw_row.pack(fill=tk.X, pady=(8, 4))
        
        StyledWidgetsHelper.create_label(webcam_draw_row, "Color: ", font_size=9, fg=TEXT_SECONDARY).pack(side=tk.LEFT)
        self.webcam_color_dropdown = ttk.Combobox(webcam_draw_row, state="readonly", width=8)
        self.webcam_color_dropdown['values'] = ["Indigo", "Emerald", "Crimson", "Amber", "Slate"]
        self.webcam_color_dropdown.current(0)  # Default: Indigo
        self.webcam_color_dropdown.pack(side=tk.LEFT, padx=(2, 8))
        self.webcam_color_dropdown.bind("<<ComboboxSelected>>", self.on_webcam_color_changed)
        
        StyledWidgetsHelper.create_label(webcam_draw_row, "Width: ", font_size=9, fg=TEXT_SECONDARY).pack(side=tk.LEFT)
        self.webcam_thickness_dropdown = ttk.Combobox(webcam_draw_row, state="readonly", width=5)
        self.webcam_thickness_dropdown['values'] = ["2px", "4px", "6px", "8px", "12px"]
        self.webcam_thickness_dropdown.current(1)  # Default: 4px
        self.webcam_thickness_dropdown.pack(side=tk.LEFT, padx=(2, 8))
        self.webcam_thickness_dropdown.bind("<<ComboboxSelected>>", self.on_webcam_thickness_changed)
        
        self.btn_clear_webcam = StyledWidgetsHelper.create_button(
            webcam_draw_row,
            "Clear Canvas",
            self.clear_webcam_drawing,
            bg=ACCENT_SECONDARY,
            hover_bg=ACCENT_SECONDARY_HOVER
        )
        self.btn_clear_webcam.pack(side=tk.LEFT, padx=2)

        # Persistence Controls Row for Webcam
        webcam_persist_row = tk.Frame(webcam_ctrl, bg=CARD_BG)
        webcam_persist_row.pack(fill=tk.X, pady=(4, 0))
        
        self.btn_save_webcam = StyledWidgetsHelper.create_button(
            webcam_persist_row,
            "Save Anchors",
            self.save_webcam_session,
            bg="#2563EB",  # Slate Blue
            hover_bg="#1D4ED8"
        )
        self.btn_save_webcam.pack(side=tk.LEFT, padx=(0, 4))
        
        self.btn_load_webcam = StyledWidgetsHelper.create_button(
            webcam_persist_row,
            "Load Anchors",
            self.load_webcam_session,
            bg="#2563EB",
            hover_bg="#1D4ED8"
        )
        self.btn_load_webcam.pack(side=tk.LEFT, padx=4)

        # Divider line
        divider = tk.Frame(controls_container, bg=BORDER_COLOR, width=1)
        divider.grid(row=0, column=0, columnspan=2, sticky="nse", padx=10)

        # B. Android IP Camera Settings Section (Right Side of Controls)
        android_ctrl = tk.Frame(controls_container, bg=CARD_BG)
        android_ctrl.grid(row=0, column=1, padx=25, sticky="nsew")

        StyledWidgetsHelper.create_label(android_ctrl, "Android Camera Stream Link", font_size=11, is_bold=True).pack(anchor=tk.W, pady=(0, 8))

        url_input_row = tk.Frame(android_ctrl, bg=CARD_BG)
        url_input_row.pack(fill=tk.X, pady=2)

        StyledWidgetsHelper.create_label(url_input_row, "IP / URL: ", font_size=9, fg=TEXT_SECONDARY).pack(side=tk.LEFT)

        self.android_url_entry = StyledWidgetsHelper.create_entry(url_input_row, width=35)
        self.android_url_entry.pack(side=tk.LEFT, padx=10)
        # Prefill default address for DroidCam/IPWebcam convenience
        self.android_url_entry.insert(0, "http://192.168.1.100:4747/video")

        # Hand Tracking Checkbox for Android Camera
        self.android_hand_tracking_var = tk.BooleanVar(value=True)
        self.chk_android_tracking = tk.Checkbutton(
            android_ctrl,
            text="Enable Hand Tracking (MediaPipe)",
            variable=self.android_hand_tracking_var,
            command=self.on_android_tracking_toggle,
            font=("Segoe UI", 9),
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            activebackground=CARD_BG,
            activeforeground=TEXT_PRIMARY,
            selectcolor=BG_COLOR,
            cursor="hand2"
        )
        self.chk_android_tracking.pack(anchor=tk.W, pady=(6, 4))

        self.btn_toggle_android = StyledWidgetsHelper.create_button(
            android_ctrl,
            "Connect Phone Camera",
            self.toggle_android,
            bg=COLOR_SUCCESS,
            hover_bg="#059669"
        )
        self.btn_toggle_android.pack(anchor=tk.W, pady=(5, 0))

        # Drawing Controls Row for Android
        android_draw_row = tk.Frame(android_ctrl, bg=CARD_BG)
        android_draw_row.pack(fill=tk.X, pady=(8, 4))
        
        StyledWidgetsHelper.create_label(android_draw_row, "Color: ", font_size=9, fg=TEXT_SECONDARY).pack(side=tk.LEFT)
        self.android_color_dropdown = ttk.Combobox(android_draw_row, state="readonly", width=8)
        self.android_color_dropdown['values'] = ["Indigo", "Emerald", "Crimson", "Amber", "Slate"]
        self.android_color_dropdown.current(0)  # Default: Indigo
        self.android_color_dropdown.pack(side=tk.LEFT, padx=(2, 8))
        self.android_color_dropdown.bind("<<ComboboxSelected>>", self.on_android_color_changed)
        
        StyledWidgetsHelper.create_label(android_draw_row, "Width: ", font_size=9, fg=TEXT_SECONDARY).pack(side=tk.LEFT)
        self.android_thickness_dropdown = ttk.Combobox(android_draw_row, state="readonly", width=5)
        self.android_thickness_dropdown['values'] = ["2px", "4px", "6px", "8px", "12px"]
        self.android_thickness_dropdown.current(1)  # Default: 4px
        self.android_thickness_dropdown.pack(side=tk.LEFT, padx=(2, 8))
        self.android_thickness_dropdown.bind("<<ComboboxSelected>>", self.on_android_thickness_changed)
        
        self.btn_clear_android = StyledWidgetsHelper.create_button(
            android_draw_row,
            "Clear Canvas",
            self.clear_android_drawing,
            bg=ACCENT_SECONDARY,
            hover_bg=ACCENT_SECONDARY_HOVER
        )
        self.btn_clear_android.pack(side=tk.LEFT, padx=2)

        # Persistence Controls Row for Android
        android_persist_row = tk.Frame(android_ctrl, bg=CARD_BG)
        android_persist_row.pack(fill=tk.X, pady=(4, 0))
        
        self.btn_save_android = StyledWidgetsHelper.create_button(
            android_persist_row,
            "Save Anchors",
            self.save_android_session,
            bg="#2563EB",  # Slate Blue
            hover_bg="#1D4ED8"
        )
        self.btn_save_android.pack(side=tk.LEFT, padx=(0, 4))
        
        self.btn_load_android = StyledWidgetsHelper.create_button(
            android_persist_row,
            "Load Anchors",
            self.load_android_session,
            bg="#2563EB",
            hover_bg="#1D4ED8"
        )
        self.btn_load_android.pack(side=tk.LEFT, padx=4)

        help_lbl = StyledWidgetsHelper.create_label(
            android_ctrl,
            "Tip: Use http://<ip>:4747/video for DroidCam. Use http://<ip>:8080/video for IP Webcam.",
            font_size=8,
            fg=TEXT_SECONDARY
        )
        help_lbl.pack(anchor=tk.W, pady=(5, 0))

    def scan_webcams_async(self):
        """
        Triggers camera index detection in a background thread to prevent GUI lockup.
        """
        self.webcam_dropdown.configure(state="disabled")
        self.btn_scan.configure(state="disabled", text="Scanning...")
        self.webcam_dropdown['values'] = ["Scanning..."]
        self.webcam_dropdown.current(0)

        def run_scan():
            available = scan_available_webcams()
            # Inject update into GUI thread
            self.root.after(0, lambda: self.on_scan_complete(available))

        threading.Thread(target=run_scan, daemon=True).start()

    def on_scan_complete(self, devices):
        """
        Updates UI elements when camera scanning completes.
        """
        self.webcam_dropdown.configure(state="readonly")
        self.btn_scan.configure(state="normal", text="Rescan Devices")
        
        if not devices:
            self.webcam_dropdown['values'] = ["No Webcams Found"]
            self.webcam_dropdown.current(0)
            self.selected_webcam_idx = None
            return

        # Format option labels
        dropdown_options = []
        for idx in devices:
            label = f"Camera {idx}"
            if idx == 0:
                label += " (Default / Laptop)"
            dropdown_options.append(label)

        self.webcam_dropdown['values'] = dropdown_options
        self.webcam_dropdown.current(0)
        self.selected_webcam_idx = devices[0]

        # If a webcam stream is running and the device list shifts, verify it remains active
        if self.webcam_stream and self.webcam_stream.running:
            # Shift the running stream if appropriate
            self.webcam_stream.change_source(self.selected_webcam_idx)

    def on_webcam_dropdown_changed(self, event):
        """
        Event handler fired when a different local webcam is selected from the dropdown.
        """
        selection = self.webcam_dropdown.get()
        if "Camera" in selection:
            # Extract digits from the dropdown text
            idx_str = ''.join(filter(str.isdigit, selection))
            if idx_str:
                self.selected_webcam_idx = int(idx_str)
                # If active, dynamically shift source
                if self.webcam_stream and self.webcam_stream.running:
                    self.webcam_stream.change_source(self.selected_webcam_idx)

    def on_webcam_tracking_toggle(self):
        """
        Handles toggling hand tracking dynamically for the local webcam.
        """
        if self.webcam_stream is not None:
            self.webcam_stream.set_hand_tracking(self.webcam_hand_tracking_var.get())

    def on_android_tracking_toggle(self):
        """
        Handles toggling hand tracking dynamically for the Android camera.
        """
        if self.android_stream is not None:
            self.android_stream.set_hand_tracking(self.android_hand_tracking_var.get())

    def toggle_webcam(self):
        """
        Starts or stops the local webcam capture thread.
        """
        if self.selected_webcam_idx is None:
            messagebox.showwarning("No Camera Detected", "Please search and select a valid local webcam device index first.")
            return

        if self.webcam_stream is None:
            # Initialize and start stream with shared drawer and fusion manager
            self.webcam_stream = CameraStream(
                source=self.selected_webcam_idx,
                name="Local Webcam",
                shared_drawer=self.shared_drawer,
                fusion_manager=self.fusion_manager,
                is_android=False
            )
            # Apply initial hand tracking state from checkbox
            self.webcam_stream.set_hand_tracking(self.webcam_hand_tracking_var.get())
            
            # Apply current drawing configurations
            color_name = self.webcam_color_dropdown.get()
            self.webcam_stream.set_drawing_color(COLOR_MAP.get(color_name, (229, 70, 79)))
            thickness_str = self.webcam_thickness_dropdown.get()
            thickness = int(''.join(filter(str.isdigit, thickness_str)))
            self.webcam_stream.set_drawing_thickness(thickness)

            self.webcam_stream.start()
            self.btn_toggle_webcam.configure(text="Stop Webcam Feed", bg=COLOR_DANGER, hover_bg="#dc2626")
        else:
            # Stop stream and teardown object
            self.webcam_stream.stop()
            self.webcam_stream = None
            self.btn_toggle_webcam.configure(text="Start Webcam Feed", bg=COLOR_SUCCESS, hover_bg="#059669")
            self.webcam_fps_lbl.configure(text="FPS: 0.0")
            self.webcam_status_lbl.configure(text="OFFLINE", fg=COLOR_DANGER)

    def toggle_android(self):
        """
        Connects or disconnects the Android camera feed.
        """
        if self.android_stream is None:
            url = self.android_url_entry.get().strip()
            if not url:
                messagebox.showerror("Missing Address", "Please enter a valid IP stream URL (e.g. HTTP/RTSP stream).")
                return
            
            # Initialize and start network stream with shared drawer and fusion manager
            self.android_stream = CameraStream(
                source=url,
                name="Android Phone",
                shared_drawer=self.shared_drawer,
                fusion_manager=self.fusion_manager,
                is_android=True
            )
            # Apply initial hand tracking state from checkbox
            self.android_stream.set_hand_tracking(self.android_hand_tracking_var.get())
            
            # Apply current drawing configurations
            color_name = self.android_color_dropdown.get()
            self.android_stream.set_drawing_color(COLOR_MAP.get(color_name, (229, 70, 79)))
            thickness_str = self.android_thickness_dropdown.get()
            thickness = int(''.join(filter(str.isdigit, thickness_str)))
            self.android_stream.set_drawing_thickness(thickness)

            self.android_stream.start()
            self.btn_toggle_android.configure(text="Disconnect Phone", bg=COLOR_DANGER, hover_bg="#dc2626")
            self.android_url_entry.configure(state="disabled")
        else:
            # Stop network stream
            self.android_stream.stop()
            self.android_stream = None
            self.btn_toggle_android.configure(text="Connect Phone Camera", bg=COLOR_SUCCESS, hover_bg="#059669")
            self.android_url_entry.configure(state="normal")
            self.android_fps_lbl.configure(text="FPS: 0.0")
            self.android_status_lbl.configure(text="OFFLINE", fg=COLOR_DANGER)

    def update_loop(self):
        """
        Periodically checks both camera streams, updates GUI display canvas,
        refreshes FPS and connection status tags. Fired every 15ms.
        """
        # 1. Update Local Webcam view
        if self.webcam_stream is not None:
            frame, connected, status, fps = self.webcam_stream.get_frame()
            
            # Update metadata labels
            self.webcam_fps_lbl.configure(text=f"FPS: {fps:.1f}")
            status_color = COLOR_SUCCESS if connected else (COLOR_WARNING if "Connecting" in status else COLOR_DANGER)
            self.webcam_status_lbl.configure(text=status.upper(), fg=status_color)

            if connected and frame is not None:
                self.render_frame_to_label(frame, self.webcam_display)
            else:
                self.render_placeholder_to_label(status, "Local Webcam", self.webcam_display)
        else:
            self.render_placeholder_to_label("OFFLINE", "Local Webcam", self.webcam_display)

        # 2. Update Android Camera view
        if self.android_stream is not None:
            frame, connected, status, fps = self.android_stream.get_frame()

            # Update metadata labels
            self.android_fps_lbl.configure(text=f"FPS: {fps:.1f}")
            status_color = COLOR_SUCCESS if connected else (COLOR_WARNING if "Connecting" in status or "Retrying" in status else COLOR_DANGER)
            self.android_status_lbl.configure(text=status.upper(), fg=status_color)

            if connected and frame is not None:
                self.render_frame_to_label(frame, self.android_display)
            else:
                self.render_placeholder_to_label(status, "Android Phone", self.android_display)
        else:
            self.render_placeholder_to_label("OFFLINE", "Android Phone", self.android_display)
        # 3. Synchronous Multi-Camera Extrinsics Calibration
        if self.webcam_stream is not None and self.android_stream is not None:
            if self.webcam_stream.connected and self.android_stream.connected:
                if self.webcam_stream.detector is not None and self.android_stream.detector is not None:
                    if self.fusion_manager.webcam_slam is None:
                        self.fusion_manager.webcam_slam = self.webcam_stream.detector.slam
                    
                    if not self.fusion_manager.calibrated:
                        fw, _, _, _ = self.webcam_stream.get_frame()
                        fa, _, _, _ = self.android_stream.get_frame()
                        dw = self.webcam_stream.latest_depth_map
                        da = self.android_stream.latest_depth_map
                        
                        if fw is not None and fa is not None:
                            self.fusion_manager.calibrate(fw, dw, fa, da)

        # Schedule next execution frame
        self.root.after(15, self.update_loop)

    def render_frame_to_label(self, frame, label_widget):
        """
        Resizes and converts an OpenCV frame (BGR array) to a Tkinter ImageTk
        object and draws it on the specified label.
        """
        try:
            # Convert frame from OpenCV (BGR) to PIL (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Compute aspect ratio fit for 480x360 window
            target_w, target_h = 480, 360
            orig_h, orig_w = frame.shape[:2]
            
            # Scale proportionally to fit 480x360 card
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Resize image
            resized_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            # Embed inside a black backdrop of exact size 480x360 to keep spacing symmetric
            backdrop = Image.new("RGB", (target_w, target_h), (15, 23, 42)) # slate-955 backdrop
            # Center image on backdrop
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            backdrop.paste(resized_image, (offset_x, offset_y))
            
            # Draw on GUI
            tk_image = ImageTk.PhotoImage(backdrop)
            label_widget.configure(image=tk_image)
            label_widget.image = tk_image  # Keep a reference to prevent garbage collection
        except Exception as e:
            # Fallback if frame processing fails
            self.render_placeholder_to_label("Display Error", "Camera Frame", label_widget)

    def render_placeholder_to_label(self, status, camera_name, label_widget):
        """
        Renders a static visual placeholder indicating stream offline states.
        """
        placeholder_tk = self.placeholder_manager.get_placeholder(status, camera_name)
        label_widget.configure(image=placeholder_tk)
        label_widget.image = placeholder_tk

    def on_webcam_color_changed(self, event):
        """Callback fired when the webcam drawing color is changed."""
        color_name = self.webcam_color_dropdown.get()
        color_bgr = COLOR_MAP.get(color_name, (229, 70, 79))
        if self.webcam_stream is not None:
            self.webcam_stream.set_drawing_color(color_bgr)

    def on_webcam_thickness_changed(self, event):
        """Callback fired when the webcam drawing thickness is changed."""
        val = self.webcam_thickness_dropdown.get()
        thickness = int(''.join(filter(str.isdigit, val)))
        if self.webcam_stream is not None:
            self.webcam_stream.set_drawing_thickness(thickness)

    def clear_webcam_drawing(self):
        """Clears the drawing path on the webcam stream."""
        if self.webcam_stream is not None:
            self.webcam_stream.clear_drawing()

    def on_android_color_changed(self, event):
        """Callback fired when the Android camera drawing color is changed."""
        color_name = self.android_color_dropdown.get()
        color_bgr = COLOR_MAP.get(color_name, (229, 70, 79))
        if self.android_stream is not None:
            self.android_stream.set_drawing_color(color_bgr)

    def on_android_thickness_changed(self, event):
        """Callback fired when the Android camera drawing thickness is changed."""
        val = self.android_thickness_dropdown.get()
        thickness = int(''.join(filter(str.isdigit, val)))
        if self.android_stream is not None:
            self.android_stream.set_drawing_thickness(thickness)

    def clear_android_drawing(self):
        """Clears the drawing path on the Android camera stream."""
        if self.android_stream is not None:
            self.android_stream.clear_drawing()

    def save_webcam_session(self):
        """Saves the local webcam SLAM map and drawings to file."""
        if self.webcam_stream is not None:
            success = self.webcam_stream.save_session("webcam_session.json", self.fusion_manager)
            if success:
                messagebox.showinfo("Anchors Persistent", "Successfully saved current 3D world anchors & drawings to webcam_session.json!")
            else:
                messagebox.showerror("Error", "Failed to save anchors session. Make sure hand tracking and camera feed are active.")
        else:
            messagebox.showwarning("Stream Offline", "Webcam stream must be running with hand tracking active to save anchors.")

    def load_webcam_session(self):
        """Loads a saved SLAM map and drawings for local webcam."""
        if self.webcam_stream is not None:
            import os
            if not os.path.exists("webcam_session.json"):
                messagebox.showwarning("No Save Found", "No saved session 'webcam_session.json' was found in the workspace.")
                return
            success = self.webcam_stream.load_session("webcam_session.json", self.fusion_manager)
            if success:
                messagebox.showinfo("Anchors Restored", "Successfully restored 3D world anchors & drawings from webcam_session.json!")
            else:
                messagebox.showerror("Error", "Failed to load anchors session. Verify the file format.")
        else:
            messagebox.showwarning("Stream Offline", "Webcam stream must be running with hand tracking active to load anchors.")

    def save_android_session(self):
        """Saves the Android phone camera SLAM map and drawings to file."""
        if self.android_stream is not None:
            success = self.android_stream.save_session("android_session.json", self.fusion_manager)
            if success:
                messagebox.showinfo("Anchors Persistent", "Successfully saved current 3D world anchors & drawings to android_session.json!")
            else:
                messagebox.showerror("Error", "Failed to save anchors session. Make sure hand tracking and camera feed are active.")
        else:
            messagebox.showwarning("Stream Offline", "Android camera stream must be running with hand tracking active to save anchors.")

    def load_android_session(self):
        """Loads a saved SLAM map and drawings for Android phone camera."""
        if self.android_stream is not None:
            import os
            if not os.path.exists("android_session.json"):
                messagebox.showwarning("No Save Found", "No saved session 'android_session.json' was found in the workspace.")
                return
            success = self.android_stream.load_session("android_session.json", self.fusion_manager)
            if success:
                messagebox.showinfo("Anchors Restored", "Successfully restored 3D world anchors & drawings from android_session.json!")
            else:
                messagebox.showerror("Error", "Failed to load anchors session. Verify the file format.")
        else:
            messagebox.showwarning("Stream Offline", "Android camera stream must be running with hand tracking active to load anchors.")

    def on_closing(self):
        """
        Clean release handler when user clicks the window exit cross.
        """
        # Stop background threads
        if self.webcam_stream is not None:
            self.webcam_stream.stop()
        if self.android_stream is not None:
            self.android_stream.stop()
            
        self.root.destroy()


if __name__ == "__main__":
    # Create the root window context
    root = tk.Tk()
    
    # Initialize Application Controller class
    app = CameraPortalApp(root)
    
    # Keep window loop active
    root.mainloop()
