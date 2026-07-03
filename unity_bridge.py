import socket
import json
import threading
import time

class UnityBridge:
    """
    Handles TCP socket communication with the Unity SlamReceiver server.
    Streams camera pose, point cloud coordinates, drawings, and shapes in real-time.
    """
    def __init__(self, host="127.0.0.1", port=9090):
        self.host = host
        self.port = port
        self.client_socket = None
        self.connected = False
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

    def start(self):
        """Starts the connection worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._connection_loop, name="UnityBridge_Thread")
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Disconnects and terminates the bridge thread."""
        self.running = False
        self.disconnect()
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def connect(self):
        """Attempts to connect to the Unity server."""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(2.0)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            print(f"[UnityBridge] Connected to Unity on {self.host}:{self.port}")
            return True
        except Exception:
            self.connected = False
            return False

    def disconnect(self):
        """Closes the current socket connection."""
        self.connected = False
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None

    def _connection_loop(self):
        """Worker thread loop that ensures connection stability and retries."""
        while self.running:
            if not self.connected:
                self.connect()
            time.sleep(2.0)  # Retry connection every 2 seconds if down

    def stream_frame_data(self, cam_pos, cam_rot_3x3, point_cloud_pts, drawing_strokes, shapes_list):
        """
        Serializes and streams the SLAM camera frame data to Unity.
        
        Args:
            cam_pos (list/tuple): [X, Y, Z] camera center in world coordinates.
            cam_rot_3x3 (np.ndarray): 3x3 rotation matrix.
            point_cloud_pts (list): List of [X, Y, Z] points in the sparse map.
            drawing_strokes (list): List of lists of [X, Y, Z] drawing points.
            shapes_list (list): List of recognized shapes (dicts with type, pos, scale).
        """
        if not self.connected or self.client_socket is None:
            return

        try:
            # Flatten rotation matrix for JSON utility compatibility
            rot_flat = cam_rot_3x3.flatten().tolist() if cam_rot_3x3 is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1]

            # Format shape data
            shapes_formatted = []
            for shape in shapes_list:
                shapes_formatted.append({
                    "type": str(shape.get("type", "unknown")),
                    "pos": [float(val) for val in shape.get("pos", [0, 0, 75])],
                    "scale": float(shape.get("scale", 5.0))
                })

            # Reformat payload to match Unity class names
            payload = {
                "cam_pos": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                "cam_rot": rot_flat,
                "point_cloud": [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in point_cloud_pts],
                "strokes": [[[float(pt[0]), float(pt[1]), float(pt[2])] for pt in stroke] for stroke in drawing_strokes],
                "shapes": shapes_formatted
            }

            # Send payload terminated with a newline character as the packet delimiter
            message = json.dumps(payload) + "\n"
            
            with self.lock:
                self.client_socket.sendall(message.encode("utf-8"))
        except (socket.error, socket.timeout):
            print("[UnityBridge] Connection lost, retrying...")
            self.disconnect()
        except Exception as e:
            print(f"[UnityBridge] Streaming error: {e}")
