import cv2
import threading
import time
import numpy as np
import torch
from hands import HandDetector

class CameraStream:
    """
    Handles capturing frames from a video source (local webcam or network stream)
    in a dedicated background thread.
    
    Features:
    - Multi-threaded frame grabbing to prevent GUI lag.
    - Thread-safe frame retrieval.
    - Real-time FPS calculation.
    - Automatic reconnection logic if the stream goes down.
    - Optional real-time hand landmark and ID tracking using MediaPipe.
    - Asynchronous relative depth estimation using Depth Anything V2.
    """
    def __init__(self, source, name="Camera", shared_drawer=None, fusion_manager=None, is_android=False):
        """
        Initializes the CameraStream.
        
        Args:
            source (int or str): Camera source. An integer (e.g. 0) representing a local webcam,
                                 or a string (url) representing an IP camera stream.
            name (str): Friendly name for the camera (e.g., "Webcam" or "Android Camera").
            shared_drawer (AirDrawer): Shared drawing canvas engine.
            fusion_manager (CameraFusionManager): Multi-camera pose fusion coordinator.
            is_android (bool): True if this stream represents the Android phone camera.
        """
        self.source = source
        self.name = name
        self.cap = None
        self.running = False
        self.connected = False
        self.fps = 0.0
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.status = "Stopped"
        
        # Hand tracking variables
        self.hand_tracking_enabled = False
        self.detector = None  # Lazily initialized in the background thread
        
        # Drawing configuration cache (applied to detector upon instantiation)
        self.drawing_color = (229, 70, 79)  # Default BGR: Indigo
        self.drawing_thickness = 4
        
        # 3D Depth estimation variables
        self.raw_frame = None
        self.latest_depth_map = None
        self.depth_estimator = None
        self.depth_thread = None
        
        # Multi-camera fusion configurations
        self.shared_drawer = shared_drawer
        self.fusion_manager = fusion_manager
        self.is_android = is_android

    def start(self):
        """
        Starts the camera capture background thread.
        """
        if self.running:
            return
        
        self.running = True
        self.status = "Connecting"
        self.thread = threading.Thread(target=self._capture_loop, name=f"{self.name}_Thread")
        self.thread.daemon = True  # Daemonize thread so it dies when the main program closes
        self.thread.start()
        
        # Start background depth estimation thread
        self.depth_thread = threading.Thread(target=self._depth_estimation_loop, name=f"{self.name}_Depth_Thread")
        self.depth_thread.daemon = True
        self.depth_thread.start()

    def set_hand_tracking(self, enabled):
        """
        Enables or disables MediaPipe hand tracking on the stream.
        
        Args:
            enabled (bool): True to enable hand tracking overlay, False to show raw feed.
        """
        with self.lock:
            self.hand_tracking_enabled = enabled

    def _capture_loop(self):
        """
        The main capture loop that runs in the background thread.
        It continuously grabs frames, runs hand tracking (if enabled),
        tracks FPS, and handles auto-reconnection.
        """
        fps_update_interval = 1.0  # Update FPS value every second
        frame_count = 0
        fps_timer = time.time()
        
        while self.running:
            # If the VideoCapture object is not initialized or has been closed, attempt connection
            if self.cap is None or not self.cap.isOpened():
                with self.lock:
                    self.connected = False
                    self.status = "Connecting..."
                
                # If the source is an integer, it's a local camera. Use DirectShow for faster initialization on Windows.
                if isinstance(self.source, int):
                    cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                else:
                    # For network URLs (DroidCam / IP Webcam)
                    cap = cv2.VideoCapture(self.source)
                
                if cap.isOpened():
                    with self.lock:
                        self.cap = cap
                        self.connected = True
                        self.status = "Connected"
                    # Reset FPS tracking timers
                    fps_timer = time.time()
                    frame_count = 0
                else:
                    # Connection failed. Release resources and wait before retrying.
                    cap.release()
                    with self.lock:
                        self.status = "Disconnected (Retrying...)"
                    # Wait 2 seconds before attempting reconnection to avoid hammering the system/network
                    time.sleep(2.0)
                    continue

            # Read the next frame from the camera stream
            success, frame = self.cap.read()
            
            if success and frame is not None:
                # Save a copy of the raw frame for the background depth estimation thread
                with self.lock:
                    self.raw_frame = frame.copy()
                    
                # Apply hand tracking overlay if enabled
                with self.lock:
                    tracking_active = self.hand_tracking_enabled
                
                if tracking_active:
                    try:
                        # Lazily initialize hand detector in background thread to prevent GUI load spikes
                        if self.detector is None:
                            self.detector = HandDetector(shared_drawer=self.shared_drawer)
                            self.detector.set_drawing_color(self.drawing_color)
                            self.detector.set_drawing_thickness(self.drawing_thickness)
                        
                        # Retrieve the latest estimated depth map from the shared buffer
                        with self.lock:
                            depth_map = self.latest_depth_map
                            
                        # Process frame to detect hands, estimate depth, and draw overlays
                        camera_name = "Android" if self.is_android else "Webcam"
                        processed_frame, _ = self.detector.process_frame(
                            frame=frame,
                            depth_map=depth_map,
                            camera_name=camera_name,
                            fusion_manager=self.fusion_manager
                        )
                        output_frame = processed_frame
                    except Exception as e:
                        # Fallback to raw frame in case of a processing error
                        output_frame = frame
                else:
                    output_frame = frame

                # Thread-safe copy of the output frame to make it accessible to the GUI
                with self.lock:
                    self.frame = output_frame.copy()
                    self.connected = True
                    self.status = "Connected"
                
                # Calculate FPS
                frame_count += 1
                curr_time = time.time()
                elapsed = curr_time - fps_timer
                if elapsed >= fps_update_interval:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    fps_timer = curr_time
            else:
                # Read failed (camera disconnected or network stream interrupted)
                with self.lock:
                    self.connected = False
                    self.status = "Disconnected (Reconnecting...)"
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                
                # Sleep briefly before the next loop iteration triggers a reconnect attempt
                time.sleep(1.0)

            # Control capture rate (helps reduce CPU usage slightly)
            time.sleep(0.01)

        # Cleanup if the thread is instructed to stop
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.connected = False
            self.status = "Stopped"

    def get_frame(self):
        """
        Retrieves the latest captured frame and current status in a thread-safe manner.
        
        Returns:
            tuple: (frame, connected, status_text, fps)
                   - frame: numpy array or None
                   - connected: bool
                   - status_text: str
                   - fps: float
        """
        with self.lock:
            return self.frame, self.connected, self.status, self.fps

    def change_source(self, new_source):
        """
        Changes the camera source and triggers a reconnect in the background loop.
        
        Args:
            new_source (int or str): The new camera source index or URL.
        """
        with self.lock:
            self.source = new_source
            self.connected = False
            self.status = "Connecting"
            # Release current capture to force the background loop to re-initialize with new source
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def stop(self):
        """
        Stops the camera capture loop and releases the camera device.
        """
        self.running = False
        # If threads exist, wait briefly for them to exit
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.depth_thread:
            self.depth_thread.join(timeout=1.0)
            self.depth_thread = None

    def clear_drawing(self):
        """
        Clears the drawing paths in the hand detector.
        """
        with self.lock:
            if self.detector is not None:
                self.detector.clear_drawing()

    def set_drawing_color(self, color_bgr):
        """
        Sets the drawing color and caches it for future hand detectors.
        """
        with self.lock:
            self.drawing_color = color_bgr
            if self.detector is not None:
                self.detector.set_drawing_color(color_bgr)

    def set_drawing_thickness(self, thickness):
        """
        Sets the drawing line thickness and caches it for future hand detectors.
        """
        with self.lock:
            self.drawing_thickness = thickness
            if self.detector is not None:
                self.detector.set_drawing_thickness(thickness)

    def save_session(self, filepath, fusion_manager=None):
        """Saves current drawing session and SLAM map database to a file."""
        with self.lock:
            if self.detector is not None:
                return self.detector.save_session(filepath, fusion_manager)
            return False

    def load_session(self, filepath, fusion_manager=None):
        """Loads a drawing session and SLAM map database from a file."""
        with self.lock:
            if self.detector is not None:
                return self.detector.load_session(filepath, fusion_manager)
            return False

    def _depth_estimation_loop(self):
        """
        Background worker thread that continually estimates relative depth 
        on the latest captured video frame using Depth Anything V2.
        """
        while self.running:
            frame_to_process = None
            with self.lock:
                # Retrieve the latest raw frame captured by the thread
                if self.connected and self.raw_frame is not None:
                    frame_to_process = self.raw_frame.copy()
            
            if frame_to_process is not None:
                try:
                    # Lazily load Depth Anything V2 pipeline in this background thread
                    if self.depth_estimator is None:
                        from transformers import pipeline
                        self.depth_estimator = pipeline(
                            task="depth-estimation",
                            model="depth-anything/Depth-Anything-V2-Small-hf",
                            device="cpu"
                        )
                    
                    # 1. Downsample image to speed up CPU inference dramatically
                    h, w = frame_to_process.shape[:2]
                    rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                    
                    from PIL import Image
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # Resize to a smaller dimension (256x192) for real-time CPU performance
                    small_img = pil_img.resize((256, 192), Image.Resampling.BILINEAR)
                    
                    # Run Depth Anything V2 inference
                    res = self.depth_estimator(small_img)
                    depth_map_pil = res["depth"]
                    
                    # Convert back to numpy array
                    depth_map = np.array(depth_map_pil)
                    
                    # Upsample depth map back to original frame size
                    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # Thread-safe cache update
                    with self.lock:
                        self.latest_depth_map = depth_map_resized
                except Exception as e:
                    # Log error silently to prevent terminal spam
                    pass
            
            # Control execution rate (~10 depth maps per second to reduce CPU usage)
            time.sleep(0.08)


def scan_available_webcams(max_to_scan=5):
    """
    Scans camera indices starting from 0 to check which webcams are available.
    
    Args:
        max_to_scan (int): The maximum index to check.
        
    Returns:
        list: A list of integers representing the active webcam indices.
    """
    available_indices = []
    for idx in range(max_to_scan):
        # Use cv2.CAP_DSHOW to prevent slow startup times on Windows
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Try to read a frame to confirm the webcam is working and accessible
            ret, frame = cap.read()
            if ret and frame is not None:
                available_indices.append(idx)
            cap.release()
    return available_indices
