import cv2
import mediapipe as mp
import numpy as np
import math
from unity_bridge import UnityBridge

class VisualSlamEngine:
    """
    Python visual SLAM engine implementing ORB feature extraction, Brute-Force matching,
    and RANSAC-based PnP camera pose tracking. Reconstructs a sparse 3D point cloud 
    map of the room and tracks the camera trajectory.
    """
    def __init__(self, max_features=800):
        """Initializes the ORB feature extractor and pose structures."""
        self.orb = cv2.ORB_create(nfeatures=max_features)
        # Brute-Force Matcher with Hamming distance for ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Camera pose: transforms world to camera coordinates
        self.R_w = np.identity(3, dtype=np.float32)
        self.t_w = np.zeros((3, 1), dtype=np.float32)
        
        # Trajectory list containing 3D coordinates (X, Y, Z) in world space
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        
        # Sparse point cloud of map points in world space
        self.map_points = []
        
        # Features cache for consecutive frame tracking
        self.last_kps = None
        self.last_des = None
        self.last_pts_3d = None  # 3D points of cached features in world coordinates
        
        self.tracking_status = "Initialized"
        
    def reset(self):
        """Resets the SLAM coordinate system, map, and trajectory."""
        self.R_w = np.identity(3, dtype=np.float32)
        self.t_w = np.zeros((3, 1), dtype=np.float32)
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        self.map_points.clear()
        self.last_kps = None
        self.last_des = None
        self.last_pts_3d = None
        self.tracking_status = "Reset"
        
    def process_frame(self, frame, depth_map):
        """Processes the frame to extract ORB features, match them, and track camera pose."""
        if frame is None:
            return
            
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Feature Detection and Description
        kps, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(kps) < 12:
            self.tracking_status = "Low Features"
            return
            
        # Define Camera Intrinsic Matrix K (Approximated)
        K = np.array([
            [w, 0, w / 2.0],
            [0, w, h / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float32)
        
        # 2. Extract 3D Camera Coordinates of current features using depth map
        pts_3d_cam = []
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            if depth_map is not None:
                d_val = depth_map[y, x]
            else:
                d_val = 128.0
                
            d_norm = d_val / 255.0
            z = 30.0 + (1.0 - d_norm) * 90.0
            
            x_c = (kp.pt[0] - w / 2.0) * (z / w)
            y_c = (kp.pt[1] - h / 2.0) * (z / w)
            
            pts_3d_cam.append(np.array([x_c, y_c, z], dtype=np.float32))
            
        # First frame initialization
        if self.last_des is None:
            self.R_w = np.identity(3, dtype=np.float32)
            self.t_w = np.zeros((3, 1), dtype=np.float32)
            
            # For frame 0, camera coordinates = world coordinates
            self.last_pts_3d = pts_3d_cam
            for pt in pts_3d_cam:
                self.map_points.append(pt)
                
            self.last_kps = kps
            self.last_des = des
            self.tracking_status = "Tracking OK"
            return
            
        # 3. Feature Matching
        matches = self.matcher.match(self.last_des, des)
        
        if len(matches) < 8:
            self.tracking_status = "Lost (Few Matches)"
            return
            
        # Compile corresponding 3D world points and 2D current image points
        obj_pts = []  # 3D points in world coordinates (from previous frame)
        img_pts = []  # 2D points in current frame
        
        for m in matches:
            prev_idx = m.queryIdx
            curr_idx = m.trainIdx
            
            obj_pts.append(self.last_pts_3d[prev_idx])
            img_pts.append(kps[curr_idx].pt)
            
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        # 4. Pose Tracking via PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=False,
            iterationsCount=100,
            reprojectionError=8.0
        )
        
        if success and inliers is not None and len(inliers) >= 8:
            R, _ = cv2.Rodrigues(rvec)
            self.R_w = R.astype(np.float32)
            self.t_w = tvec.astype(np.float32)
            
            # Camera Optical Center in world coordinates: C = -R^T * t
            camera_center = -np.dot(self.R_w.T, self.t_w).flatten()
            self.trajectory.append(camera_center)
            
            # Convert current 3D camera coordinates to 3D world coordinates for the next frame
            # P_w = R^T * (P_c - t)
            self.last_pts_3d = []
            for i in range(len(kps)):
                pt_cam = pts_3d_cam[i]
                pt_world = np.dot(self.R_w.T, pt_cam.reshape(3, 1) - self.t_w).flatten()
                self.last_pts_3d.append(pt_world)
                
                # Sparsely add points to the point cloud map to keep size bounded
                if i % 15 == 0:
                    self.map_points.append(pt_world)
            
            # Bound point cloud size to prevent performance decay
            if len(self.map_points) > 1200:
                self.map_points = self.map_points[-900:]
                
            self.last_kps = kps
            self.last_des = des
            self.tracking_status = "Tracking OK"
        else:
            self.tracking_status = "Lost (RANSAC Fail)"

    def transform_cam_to_world(self, pt_c):
        """Transforms 3D Camera coordinates to 3D World coordinates."""
        # P_w = R^T * (P_c - t)
        pt_c_arr = np.array(pt_c, dtype=np.float32).reshape(3, 1)
        pt_w = np.dot(self.R_w.T, pt_c_arr - self.t_w).flatten()
        return (pt_w[0], pt_w[1], pt_w[2])
        
    def transform_world_to_cam(self, pt_w):
        """Transforms 3D World coordinates to 3D Camera coordinates."""
        # P_c = R * P_w + t
        pt_w_arr = np.array(pt_w, dtype=np.float32).reshape(3, 1)
        pt_c = np.dot(self.R_w, pt_w_arr) + self.t_w
        pt_c = pt_c.flatten()
        return (pt_c[0], pt_c[1], pt_c[2])


class CameraFusionManager:
    """
    Manages multi-camera calibration, coordinate alignment, and observation fusion
    between the local webcam and the Android IP camera.
    """
    def __init__(self, shared_drawer=None):
        self.drawer = shared_drawer
        
        # Relative transformation: transforms Android camera coordinates to Webcam coordinates
        self.R_android_to_webcam = np.identity(3, dtype=np.float32)
        self.t_android_to_webcam = np.zeros((3, 1), dtype=np.float32)
        self.calibrated = False
        
        # Reference to the main webcam SLAM engine (for shared world coordinates)
        self.webcam_slam = None
        
        # ORB matcher for calibration
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def calibrate(self, frame_webcam, depth_webcam, frame_android, depth_android):
        """
        Calibrates the relative pose between the Android camera and Webcam.
        Finds R and t that transform Android camera space into Webcam camera space.
        """
        if frame_webcam is None or frame_android is None:
            return False
            
        try:
            gray_w = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2GRAY)
            gray_a = cv2.cvtColor(frame_android, cv2.COLOR_BGR2GRAY)
            
            kps_w, des_w = self.orb.detectAndCompute(gray_w, None)
            kps_a, des_a = self.orb.detectAndCompute(gray_a, None)
            
            if des_w is None or des_a is None or len(kps_w) < 15 or len(kps_a) < 15:
                return False
                
            matches = self.matcher.match(des_w, des_a)
            if len(matches) < 15:
                return False
                
            # Get matching 2D coordinates
            pts_w_2d = []
            pts_a_2d = []
            for m in matches:
                pts_w_2d.append(kps_w[m.queryIdx].pt)
                pts_a_2d.append(kps_a[m.trainIdx].pt)
                
            # Calculate 3D coordinates in both camera spaces using their depth maps
            pts_w_3d = []
            pts_a_3d = []
            
            ww, hw = frame_webcam.shape[1], frame_webcam.shape[0]
            wa, ha = frame_android.shape[1], frame_android.shape[0]
            
            for i in range(len(matches)):
                # Webcam 3D coordinate
                pt_w = pts_w_2d[i]
                xw, yw = int(pt_w[0]), int(pt_w[1])
                xw = max(0, min(ww - 1, xw))
                yw = max(0, min(hw - 1, yw))
                dw_val = depth_webcam[yw, xw] if depth_webcam is not None else 128.0
                zw = 30.0 + (1.0 - dw_val / 255.0) * 90.0
                xwc = (pt_w[0] - ww / 2.0) * (zw / ww)
                ywc = (pt_w[1] - hw / 2.0) * (zw / ww)
                
                # Android 3D coordinate
                pt_a = pts_a_2d[i]
                xa, ya = int(pt_a[0]), int(pt_a[1])
                xa = max(0, min(wa - 1, xa))
                ya = max(0, min(ha - 1, ya))
                da_val = depth_android[ya, xa] if depth_android is not None else 128.0
                za = 30.0 + (1.0 - da_val / 255.0) * 90.0
                xac = (pt_a[0] - wa / 2.0) * (za / wa)
                yac = (pt_a[1] - ha / 2.0) * (za / wa)
                
                pts_w_3d.append([xwc, ywc, zw])
                pts_a_3d.append([xac, yac, za])
                
            pts_w_3d = np.array(pts_w_3d, dtype=np.float32)
            pts_a_3d = np.array(pts_a_3d, dtype=np.float32)
            
            # Estimate relative pose using solvePnPRansac (Android 3D points mapped to Webcam 2D pixels)
            K_w = np.array([
                [ww, 0, ww / 2.0],
                [0, ww, hw / 2.0],
                [0, 0, 1.0]
            ], dtype=np.float32)
            
            img_pts_webcam = np.array(pts_w_2d, dtype=np.float32)
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_a_3d, img_pts_webcam, K_w, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=100,
                reprojectionError=8.0
            )
            
            if success and inliers is not None and len(inliers) >= 8:
                R, _ = cv2.Rodrigues(rvec)
                self.R_android_to_webcam = R.astype(np.float32)
                self.t_android_to_webcam = tvec.astype(np.float32)
                self.calibrated = True
                return True
        except Exception:
            pass
        return False
        
    def transform_android_to_webcam(self, pt_a):
        """Transforms 3D coordinates from Android camera frame to Webcam camera frame."""
        if not self.calibrated:
            return pt_a
        pt_a_arr = np.array(pt_a, dtype=np.float32).reshape(3, 1)
        pt_w = np.dot(self.R_android_to_webcam, pt_a_arr) + self.t_android_to_webcam
        pt_w = pt_w.flatten()
        return (pt_w[0], pt_w[1], pt_w[2])


class AirDrawer:
    """
    Manages the 3D air-drawing state, stroke lists, real-time temporal/spatial smoothing,
    pinch detection, coordinate back-projection, and miniature 3D visualization.
    """
    def __init__(self, pinch_threshold=0.35, smoothing_factor=0.3):
        """
        Initializes the AirDrawer.
        
        Args:
            pinch_threshold (float): Relative distance ratio between thumb and index tip to trigger drawing.
            smoothing_factor (float): Exponential Moving Average (EMA) factor [0.0, 1.0] for coordinate smoothing.
        """
        self.pinch_threshold = pinch_threshold
        self.smoothing_factor = smoothing_factor
        
        # All completed strokes. Each stroke is a list of (X, Y, Z) physical coordinates in cm.
        self.strokes = []
        
        # Currently active strokes per hand label ("Left" and "Right")
        self.active_strokes = {
            "Left": [],
            "Right": []
        }
        
        # Last smoothed 3D coordinate per hand label for EMA calculation
        self.smoothed_cursors = {
            "Left": None,
            "Right": None
        }
        
        # Drawing configuration
        self.color = (229, 70, 79)  # Default BGR: Indigo
        self.thickness = 4
        self.recognized_shapes = []
        
        # Multi-camera observation cache for fusion
        self.last_cursor_updates = {
            "Webcam": {"Left": None, "Right": None},
            "Android": {"Left": None, "Right": None}
        }
        
    def clear(self):
        """Clears all drawn strokes and resets active drawing state."""
        self.strokes.clear()
        self.active_strokes["Left"].clear()
        self.active_strokes["Right"].clear()
        self.smoothed_cursors["Left"] = None
        self.smoothed_cursors["Right"] = None
        self.recognized_shapes.clear()
        self.last_cursor_updates = {
            "Webcam": {"Left": None, "Right": None},
            "Android": {"Left": None, "Right": None}
        }

    def recognize_and_replace_stroke(self, stroke):
        """
        Fits a 3D plane to the stroke using PCA (Principal Component Analysis),
        classifies the 2D projected shape, and returns a perfect 3D wireframe shape
        on the fitted plane if recognized.
        """
        if len(stroke) < 15:
            return stroke  # Too short to reliably classify
            
        pts = np.array(stroke, dtype=np.float32)
        n_pts = len(pts)
        
        # 1. Fit a 3D plane using PCA
        # Center the data
        center = np.mean(pts, axis=0)
        centered = pts - center
        
        # Compute 3x3 Covariance Matrix
        cov = np.dot(centered.T, centered) / n_pts
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort eigenvectors in descending order of eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Two main orthogonal directions spanning the best-fitting 3D plane
        u_axis = eigenvectors[:, 0]
        v_axis = eigenvectors[:, 1]
        
        # Project 3D points onto this 2D plane coordinate system
        pts_2d = []
        for pt in centered:
            x_2d = np.dot(pt, u_axis)
            y_2d = np.dot(pt, v_axis)
            pts_2d.append([x_2d, y_2d])
            
        pts_2d = np.array(pts_2d, dtype=np.float32)
        
        # 2. Classify 2D shape using geometric heuristics
        shape_type, param1, param2 = self._classify_shape_2d(pts_2d)
        
        if shape_type == "unknown":
            return stroke  # Fallback to rough hand drawing
            
        perfect_pts_2d = []
        
        if shape_type == "circle":
            c_2d, radius = param1, param2
            # Generate perfect circle points (40 segments)
            for i in range(41):
                theta = 2 * np.pi * i / 40.0
                px = c_2d[0] + radius * np.cos(theta)
                py = c_2d[1] + radius * np.sin(theta)
                perfect_pts_2d.append([px, py])
                
        elif shape_type == "triangle":
            approx = param1
            for v in approx:
                perfect_pts_2d.append([v[0][0], v[0][1]])
            # Close the triangle loop
            perfect_pts_2d.append([approx[0][0][0], approx[0][0][1]])
            
        elif shape_type == "square" or shape_type == "rectangle":
            rect = param1
            center_rect, size, angle = rect
            w_box, h_box = size
            
            if shape_type == "square":
                side = (w_box + h_box) / 2.0
                w_box, h_box = side, side
                
            # Get 4 corners of the rotated bounding box
            box_2d = cv2.boxPoints(((center_rect[0], center_rect[1]), (w_box, h_box), angle))
            for pt in box_2d:
                perfect_pts_2d.append([pt[0], pt[1]])
            # Close the rectangle loop
            perfect_pts_2d.append([box_2d[0][0], box_2d[0][1]])
            
        elif shape_type == "star":
            c_2d, dists = param1, param2
            # Find peaks (outer tips) and valleys (inner corners) to estimate radii
            peaks = []
            valleys = []
            for i in range(n_pts):
                prev_d = dists[(i - 1) % n_pts]
                curr_d = dists[i]
                next_d = dists[(i + 1) % n_pts]
                if curr_d >= prev_d and curr_d >= next_d:
                    peaks.append(curr_d)
                elif curr_d <= prev_d and curr_d <= next_d:
                    valleys.append(curr_d)
            
            r_out = np.mean(peaks) if peaks else np.max(dists)
            r_in = np.mean(valleys) if valleys else r_out * 0.4
            
            # Clamp inner radius to prevent degenerate star shapes
            if r_in > r_out * 0.7:
                r_in = r_out * 0.4
                
            # Generate perfect 5-pointed star coordinates (10 alternating points)
            for i in range(11):
                angle = i * np.pi / 5.0 - np.pi / 2.0
                r = r_out if i % 2 == 0 else r_in
                px = c_2d[0] + r * np.cos(angle)
                py = c_2d[1] + r * np.sin(angle)
                perfect_pts_2d.append([px, py])
                
        elif shape_type == "arrow":
            approx = param1
            # Simplify to clean straight-edged arrow segments
            for v in approx:
                perfect_pts_2d.append([v[0][0], v[0][1]])
            perfect_pts_2d.append([approx[0][0][0], approx[0][0][1]])
            
        # 3. Reconstruct 3D points from perfect 2D coordinates back onto the fitted 3D plane
        perfect_pts_3d = []
        for pt in perfect_pts_2d:
            pt_3d = center + pt[0] * u_axis + pt[1] * v_axis
            perfect_pts_3d.append((pt_3d[0], pt_3d[1], pt_3d[2]))
            
        # Append recognized 3D shape primitives for Unity rendering
        if shape_type in ["circle", "square", "rectangle", "triangle"]:
            if shape_type == "circle":
                c_2d = param1
                c_3d = center + c_2d[0] * u_axis + c_2d[1] * v_axis
                scale = float(param2)
                self.recognized_shapes.append({"type": "sphere", "pos": (float(c_3d[0]), float(c_3d[1]), float(c_3d[2])), "scale": scale})
            elif shape_type in ["square", "rectangle"]:
                c_2d, size, angle = param1
                c_3d = center + c_2d[0] * u_axis + c_2d[1] * v_axis
                scale = float((size[0] + size[1]) / 2.0)
                self.recognized_shapes.append({"type": "cube", "pos": (float(c_3d[0]), float(c_3d[1]), float(c_3d[2])), "scale": scale})
            elif shape_type == "triangle":
                c_3d = center
                scale = float(np.mean(np.linalg.norm(pts_2d, axis=1)))
                self.recognized_shapes.append({"type": "cylinder", "pos": (float(c_3d[0]), float(c_3d[1]), float(c_3d[2])), "scale": scale})
            
        return perfect_pts_3d

    def _classify_shape_2d(self, pts):
        contour = pts.astype(np.float32)
        
        # Calculate key geometric properties
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Circularity measure
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        
        # Simplify contour structure using polygon approximation
        epsilon = 0.032 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        n_vertices = len(approx)
        
        # Standard deviation of distance to center (variance check for circles)
        center = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        dist_ratio = std_dist / mean_dist if mean_dist > 0 else 1.0
        
        # 1. Detect Circles
        if dist_ratio < 0.12 or circularity > 0.82:
            return "circle", center, mean_dist
            
        # 2. Detect Triangles
        if n_vertices == 3:
            return "triangle", approx, None
            
        # 3. Detect Squares / Rectangles
        if n_vertices == 4:
            rect = cv2.minAreaRect(contour)
            w_box, h_box = rect[1]
            aspect_ratio = max(w_box, h_box) / max(min(w_box, h_box), 1e-5)
            if aspect_ratio < 1.15:
                return "square", rect, None
            else:
                return "rectangle", rect, None
                
        # 4. Detect Stars (Low solidity with 5 distinct radial extrema peaks)
        if solidity < 0.70:
            peaks = 0
            n_points = len(pts)
            for i in range(n_points):
                prev_d = dists[(i - 1) % n_points]
                curr_d = dists[i]
                next_d = dists[(i + 1) % n_points]
                if curr_d > prev_d and curr_d > next_d:
                    peaks += 1
            if peaks == 5 or n_vertices in [10, 8, 9, 11, 12]:
                return "star", center, dists
                
        # 5. Detect Arrows (Typically 7 vertices or medium solidity arrow template)
        if n_vertices == 7 or (0.68 < solidity < 0.86 and n_vertices in [5, 6, 7, 8]):
            return "arrow", approx, None
            
        # Poly fallback for noisy curves representing rectangles/squares
        if n_vertices >= 5:
            rect = cv2.minAreaRect(contour)
            w_box, h_box = rect[1]
            aspect_ratio = max(w_box, h_box) / max(min(w_box, h_box), 1e-5)
            if aspect_ratio < 1.15:
                return "square", rect, None
            else:
                return "rectangle", rect, None
                
        return "unknown", None, None

    def process_hand(self, hand_landmarks, hand_label, frame_width, frame_height, depth_map=None, slam=None, camera_name="Webcam", confidence=1.0, fusion_manager=None):
        """
        Checks for pinch gesture on a hand, queries monocular depth,
        converts 2D coordinates to 3D, transforms to World coordinates, and updates drawing state.
        Fuses multi-camera coordinates when both cameras track.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object.
            hand_label (str): "Left" or "Right"
            frame_width (int): Frame width.
            frame_height (int): Frame height.
            depth_map (numpy.ndarray): OpenCV depth map of frame shape (H, W).
            slam (VisualSlamEngine): SLAM engine representing camera pose.
            camera_name (str): "Webcam" or "Android" identifying the source camera.
            confidence (float): Hand tracking confidence score.
            fusion_manager (CameraFusionManager): Multi-camera pose fusion manager.
            
        Returns:
            tuple: (smoothed_cursor_world, is_pinching)
                   - smoothed_cursor_world: (X, Y, Z) physical world coordinates in cm.
                   - is_pinching: True if currently pinching/drawing.
        """
        # Landmarks: 0 = Wrist, 4 = Thumb Tip, 5 = Index MCP, 8 = Index Tip
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_mcp = hand_landmarks.landmark[5]
        index_tip = hand_landmarks.landmark[8]
        
        # 1. Scale-Invariant Pinch Detection Algorithm
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        dist = (dx**2 + dy**2)**0.5
        
        ref_dx = wrist.x - index_mcp.x
        ref_dy = wrist.y - index_mcp.y
        ref_dist = (ref_dx**2 + ref_dy**2)**0.5
        
        ratio = dist / max(ref_dist, 1e-5)
        is_pinching = ratio < self.pinch_threshold
        
        # Convert index tip normalized coordinates to pixel coordinates
        raw_x = int(index_tip.x * frame_width)
        raw_y = int(index_tip.y * frame_height)
        raw_x = max(0, min(frame_width - 1, raw_x))
        raw_y = max(0, min(frame_height - 1, raw_y))
        
        # 2. Monocular Depth Estimation Integration
        # Look up depth value from Depth Anything V2 (3x3 average around the coordinates to reduce noise)
        if depth_map is not None:
            y_min = max(0, raw_y - 1)
            y_max = min(frame_height - 1, raw_y + 1)
            x_min = max(0, raw_x - 1)
            x_max = min(frame_width - 1, raw_x + 1)
            d_val = np.mean(depth_map[y_min:y_max+1, x_min:x_max+1])
        else:
            d_val = 128.0  # Default mid relative depth
            
        # Convert relative disparity d_val [0, 255] to physical depth Z in cm (30cm to 120cm)
        d_norm = d_val / 255.0
        z_c = 30.0 + (1.0 - d_norm) * 90.0
        
        # 3. 2D to 3D Camera Coordinate Conversion
        # Back-project 2D pixels to 3D camera space coordinates (focal length approx = frame_width)
        x_c = (raw_x - frame_width / 2.0) * (z_c / frame_width)
        y_c = (raw_y - frame_height / 2.0) * (z_c / frame_width)
        
        # 4. Multi-Camera Extrinsics Alignment
        pt_c = (x_c, y_c, z_c)
        if camera_name == "Android" and fusion_manager is not None and fusion_manager.calibrated:
            # Transform Android camera coordinates to Webcam camera coordinates
            pt_c_webcam = fusion_manager.transform_android_to_webcam(pt_c)
            # Use Webcam's SLAM to convert to world coordinates
            if fusion_manager.webcam_slam is not None and fusion_manager.webcam_slam.tracking_status == "Tracking OK":
                pt_w = fusion_manager.webcam_slam.transform_cam_to_world(pt_c_webcam)
            else:
                pt_w = pt_c_webcam
        else:
            # Local Webcam coordinate conversion using local SLAM
            if slam is not None and slam.tracking_status == "Tracking OK":
                pt_w = slam.transform_cam_to_world(pt_c)
            else:
                pt_w = pt_c
        
        # 5. Temporal Smoothing (EMA) in World space coordinates
        prev_smooth = self.smoothed_cursors[hand_label]
        if prev_smooth is None:
            smooth_x = pt_w[0]
            smooth_y = pt_w[1]
            smooth_z = pt_w[2]
        else:
            smooth_x = self.smoothing_factor * pt_w[0] + (1.0 - self.smoothing_factor) * prev_smooth[0]
            smooth_y = self.smoothing_factor * pt_w[1] + (1.0 - self.smoothing_factor) * prev_smooth[1]
            smooth_z = self.smoothing_factor * pt_w[2] + (1.0 - self.smoothing_factor) * prev_smooth[2]
            
        smoothed_point_world = (smooth_x, smooth_y, smooth_z)
        
        # 6. Multi-Camera Observation Fusion & Best Camera Selection
        import time
        self.last_cursor_updates[camera_name][hand_label] = {
            "pt": smoothed_point_world,
            "conf": confidence,
            "time": time.time()
        }
        
        other_cam = "Android" if camera_name == "Webcam" else "Webcam"
        other_update = self.last_cursor_updates[other_cam][hand_label]
        
        fused_point_world = smoothed_point_world
        
        if other_update is not None and (time.time() - other_update["time"]) < 0.05:
            # Both cameras are actively tracking: Perform weighted fusion
            c1 = confidence
            c2 = other_update["conf"]
            if c1 + c2 > 0:
                p1 = np.array(smoothed_point_world)
                p2 = np.array(other_update["pt"])
                fused_arr = (c1 * p1 + c2 * p2) / (c1 + c2)
                fused_point_world = (fused_arr[0], fused_arr[1], fused_arr[2])
        else:
            # Only one camera is active, or other camera is stale: use current stream (implicit selection)
            pass
            
        self.smoothed_cursors[hand_label] = fused_point_world
        
        # 7. Stroke Management
        if is_pinching:
            self.active_strokes[hand_label].append(fused_point_world)
        else:
            # If pinch is released, finalize the current active stroke if it has points
            if self.active_strokes[hand_label]:
                raw_stroke = self.active_strokes[hand_label]
                perfect_stroke = self.recognize_and_replace_stroke(raw_stroke)
                self.strokes.append(perfect_stroke)
                self.active_strokes[hand_label].clear()
                
        return fused_point_world, is_pinching

    def finalize_hand(self, hand_label):
        """Finalizes the active stroke when a hand is no longer detected."""
        if self.active_strokes[hand_label]:
            raw_stroke = self.active_strokes[hand_label]
            perfect_stroke = self.recognize_and_replace_stroke(raw_stroke)
            self.strokes.append(perfect_stroke)
            self.active_strokes[hand_label].clear()
        self.smoothed_cursors[hand_label] = None

    def draw_on_frame(self, frame, slam=None):
        """Draws all completed and active strokes, alongside hover cursors and the 3D viewport."""
        h, w, _ = frame.shape
        
        # Draw completed strokes projected from World to current Camera space
        for stroke in self.strokes:
            self._draw_stroke_3d(frame, stroke, slam)
            
        # Draw active strokes for both hands
        for label, stroke in self.active_strokes.items():
            if stroke:
                self._draw_stroke_3d(frame, stroke, slam)
                
            # Draw visual hover/draw cursor at index tip (projected from 3D to 2D)
            cursor_3d_world = self.smoothed_cursors[label]
            if cursor_3d_world is not None:
                is_drawing = len(stroke) > 0
                
                # Transform from World coordinates back to current Camera coordinates for rendering
                if slam is not None and slam.tracking_status == "Tracking OK":
                    x_3d, y_3d, z_3d = slam.transform_world_to_cam(cursor_3d_world)
                else:
                    x_3d, y_3d, z_3d = cursor_3d_world
                    
                px = int(x_3d * w / z_3d + w / 2.0)
                py = int(y_3d * w / z_3d + h / 2.0)
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                
                # Apply depth-based thickness scaling
                curr_thickness = max(2, int(self.thickness * (60.0 / z_3d)))
                
                if is_drawing:
                    cv2.circle(frame, (px, py), curr_thickness + 2, self.color, -1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), curr_thickness + 3, (120, 120, 120), 1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 2, (120, 120, 120), -1, lineType=cv2.LINE_AA)

        # Draw 3D Viewport in bottom-right corner with SLAM parameters (trajectory + points)
        self._draw_3d_viewport(frame, slam)

    def _draw_stroke_3d(self, frame, stroke, slam=None):
        """Draws a 3D stroke using 3D Bezier Curve interpolation projected back to 2D using current camera pose."""
        if not stroke:
            return
            
        h, w, _ = frame.shape
        n_points = len(stroke)
        
        def project_and_draw_point(pt_w, thickness_scale=1.0):
            # Transform from World coordinates to current Camera coordinates
            if slam is not None and slam.tracking_status == "Tracking OK":
                x_3d, y_3d, z_3d = slam.transform_world_to_cam(pt_w)
            else:
                x_3d, y_3d, z_3d = pt_w
                
            px = int(x_3d * w / z_3d + w / 2.0)
            py = int(y_3d * w / z_3d + h / 2.0)
            
            # Depth visual cue: scale thickness and shade color based on depth
            thickness = max(1, int(self.thickness * (60.0 / z_3d) * thickness_scale))
            factor = max(0.0, min(1.0, (120.0 - z_3d) / 90.0))
            
            b, g, r = self.color
            shaded_color = (
                int(b * (0.3 + 0.7 * factor)),
                int(g * (0.3 + 0.7 * factor)),
                int(r * (0.3 + 0.7 * factor))
            )
            return (px, py), thickness, shaded_color

        if n_points == 1:
            coords, thickness, color = project_and_draw_point(stroke[0], thickness_scale=0.5)
            cv2.circle(frame, coords, thickness, color, -1, lineType=cv2.LINE_AA)
        elif n_points == 2:
            coords1, thickness1, color1 = project_and_draw_point(stroke[0])
            coords2, _, _ = project_and_draw_point(stroke[1])
            cv2.line(frame, coords1, coords2, color1, thickness1, lineType=cv2.LINE_AA)
        else:
            # 3D Midpoint Quadratic Bezier Curves
            def midpoint_3d(p1, p2):
                return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
                
            # Draw start segment
            mid_first = midpoint_3d(stroke[0], stroke[1])
            coords0, thickness0, color0 = project_and_draw_point(stroke[0])
            coords_mid, _, _ = project_and_draw_point(mid_first)
            cv2.line(frame, coords0, coords_mid, color0, thickness0, lineType=cv2.LINE_AA)
            
            # Draw Bezier segments between midpoints
            for i in range(1, n_points - 1):
                p_start = midpoint_3d(stroke[i-1], stroke[i])
                p_control = stroke[i]
                p_end = midpoint_3d(stroke[i], stroke[i+1])
                
                # Sample the quadratic Bezier curve in 3D space
                num_segments = 6
                pts_curve = []
                for j in range(num_segments + 1):
                    t = j / num_segments
                    x = (1 - t)**2 * p_start[0] + 2 * (1 - t) * t * p_control[0] + t**2 * p_end[0]
                    y = (1 - t)**2 * p_start[1] + 2 * (1 - t) * t * p_control[1] + t**2 * p_end[1]
                    z = (1 - t)**2 * p_start[2] + 2 * (1 - t) * t * p_control[2] + t**2 * p_end[2]
                    pts_curve.append((x, y, z))
                
                for j in range(len(pts_curve) - 1):
                    coords_c1, thickness_c1, color_c1 = project_and_draw_point(pts_curve[j])
                    coords_c2, _, _ = project_and_draw_point(pts_curve[j+1])
                    cv2.line(frame, coords_c1, coords_c2, color_c1, thickness_c1, lineType=cv2.LINE_AA)
                    
            # Draw end segment
            mid_last = midpoint_3d(stroke[-2], stroke[-1])
            coords_mid_last, thickness_mid, color_mid = project_and_draw_point(mid_last)
            coords_end, _, _ = project_and_draw_point(stroke[-1])
            cv2.line(frame, coords_mid_last, coords_end, color_mid, thickness_mid, lineType=cv2.LINE_AA)

    def _draw_3d_viewport(self, frame, slam=None):
        """Draws a miniature 3D viewport displaying trajectory, point cloud, and world-space strokes."""
        h, w, _ = frame.shape
        
        # Viewport position and size
        vx, vy = w - 85, h - 85
        
        # Semi-transparent dark background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 160, h - 160), (w - 10, h - 10), (30, 41, 59), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.rectangle(frame, (w - 160, h - 160), (w - 10, h - 10), (71, 85, 105), 1, lineType=cv2.LINE_AA)
        
        # Rotation angles: 45 deg Y-axis, 20 deg X-axis
        theta = math.radians(45)
        phi = math.radians(20)
        
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        
        def project(pt):
            # Normalize coordinates: X in [-30, 30], Y in [-30, 30], Z in [30, 120]
            xn = pt[0] / 30.0
            yn = pt[1] / 30.0
            zn = (pt[2] - 75.0) / 45.0
            
            # Rotate around Y-axis
            x1 = xn * cos_t + zn * sin_t
            y1 = yn
            z1 = -xn * sin_t + zn * cos_t
            
            # Rotate around X-axis
            xr = x1
            yr = y1 * cos_p - z1 * sin_p
            
            # Orthographic project to viewport (scale factor = 45 pixels)
            sx = int(vx + xr * 45)
            sy = int(vy - yr * 45)  # Subtract because screen Y increases downward
            return (sx, sy)

        # Draw coordinate axes (Origin at (0, 0, 75.0) in physical space)
        p_org = project((0, 0, 75.0))
        p_x = project((15.0, 0, 75.0))
        p_y = project((0, -15.0, 75.0))
        p_z = project((0, 0, 105.0))
        
        cv2.line(frame, p_org, p_x, (0, 0, 255), 1, lineType=cv2.LINE_AA)  # X Axis (Red)
        cv2.line(frame, p_org, p_y, (0, 255, 0), 1, lineType=cv2.LINE_AA)  # Y Axis (Green)
        cv2.line(frame, p_org, p_z, (255, 0, 0), 1, lineType=cv2.LINE_AA)  # Z Axis (Blue)
        
        cv2.putText(frame, "X", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Y", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Z", p_z, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        
        title_text = "3D Map View"
        if slam is not None:
            title_text = f"3D Map ({slam.tracking_status})"
        cv2.putText(frame, title_text, (w - 150, h - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

        # Draw SLAM point cloud map if available
        if slam is not None and slam.map_points:
            for pt in slam.map_points:
                sx, sy = project(pt)
                if w - 160 < sx < w - 10 and h - 160 < sy < h - 10:
                    cv2.circle(frame, (sx, sy), 1, (100, 116, 139), -1)  # Slate point cloud dots

        # Draw SLAM camera trajectory if available
        if slam is not None and len(slam.trajectory) >= 2:
            pts_traj = [project(pt) for pt in slam.trajectory]
            for i in range(len(pts_traj) - 1):
                p1, p2 = pts_traj[i], pts_traj[i+1]
                if w - 160 < p1[0] < w - 10 and h - 160 < p1[1] < h - 10 and w - 160 < p2[0] < w - 10 and h - 160 < p2[1] < h - 10:
                    cv2.line(frame, p1, p2, (34, 197, 94), 1, lineType=cv2.LINE_AA)  # Green trajectory line
                    
            # Draw current camera position marker and orientation vector in green
            cam_pos = slam.trajectory[-1]
            p_cam_center = project(cam_pos)
            
            # Draw optical center dir (15cm forward relative to camera center)
            opt_axis_world = slam.transform_cam_to_world((0, 0, 15.0))
            p_cam_dir = project(opt_axis_world)
            
            if w - 160 < p_cam_center[0] < w - 10 and h - 160 < p_cam_center[1] < h - 10:
                cv2.circle(frame, p_cam_center, 3, (16, 185, 129), -1)
                if w - 160 < p_cam_dir[0] < w - 10 and h - 160 < p_cam_dir[1] < h - 10:
                    cv2.line(frame, p_cam_center, p_cam_dir, (16, 185, 129), 1, lineType=cv2.LINE_AA)

        # Draw completed and active strokes in viewport
        for stroke in self.strokes:
            self._draw_stroke_viewport(frame, stroke, project)
            
        for stroke in self.active_strokes.values():
            if stroke:
                self._draw_stroke_viewport(frame, stroke, project)

    def _draw_stroke_viewport(self, frame, stroke, project_fn):
        if len(stroke) < 2:
            return
        pts_2d = [project_fn(pt) for pt in stroke]
        for i in range(len(pts_2d) - 1):
            cv2.line(frame, pts_2d[i], pts_2d[i+1], self.color, 1, lineType=cv2.LINE_AA)


class HandDetector:
    """
    Reusable class that leverages MediaPipe Hands to process video frames,
    detect hands, track landmarks in real-time, and annotate frames.
    Supports sharing drawers for multi-camera coordinate fusion.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, shared_drawer=None):
        """
        Initializes the MediaPipe Hands model and Visual SLAM engine.
        
        Args:
            max_num_hands (int): Maximum number of hands to detect. Default is 2.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand detection to be successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for tracking hand landmarks.
            shared_drawer (AirDrawer): Shared drawing canvas instance.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Instantiate the Visual SLAM engine
        self.slam = VisualSlamEngine()
        
        # Instantiate or assign the shared air-drawing engine
        self.drawer = shared_drawer if shared_drawer is not None else AirDrawer()
        
        # Initialize Unity Bridge socket client and start background connection thread
        self.unity_bridge = UnityBridge()
        self.unity_bridge.start()
        
        # Customize landmark and connection styles to match our application theme (indigo & emerald)
        # Note: OpenCV expects colors in BGR format.
        self.connection_spec = self.mp_draw.DrawingSpec(color=(229, 70, 79), thickness=2, circle_radius=2) # Indigo connections (BGR: 229, 70, 79)
        self.landmark_spec = self.mp_draw.DrawingSpec(color=(129, 185, 16), thickness=2, circle_radius=2)   # Emerald landmarks (BGR: 129, 185, 16)

    def process_frame(self, frame, depth_map=None, camera_name="Webcam", fusion_manager=None):
        """
        Processes a BGR image frame, runs visual SLAM pose tracking, detects hands, draws landmarks,
        overlays landmark IDs, displays classification confidence, and updates the 3D drawing engine.
        
        Args:
            frame (numpy.ndarray): OpenCV BGR frame.
            depth_map (numpy.ndarray): Optional Depth map of frame size.
            camera_name (str): "Webcam" or "Android" identifying the stream source.
            fusion_manager (CameraFusionManager): Multi-camera calibration and fusion coordinator.
            
        Returns:
            tuple: (annotated_frame, list_of_hands_metadata)
                   - annotated_frame: Frame with drawn hand overlays.
                   - list_of_hands_metadata: Lists containing labels (Left/Right) and confidence scores.
        """
        if frame is None:
            return None, []

        # 1. Run visual SLAM tracking on this frame (only on main Webcam stream to avoid CPU bottleneck)
        if camera_name == "Webcam":
            self.slam.process_frame(frame, depth_map)

        # Convert OpenCV BGR frame to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        hands_info = []

        # Draw SLAM Pose Tracking Status overlay at the top-left of the frame
        if camera_name == "Webcam":
            slam_status = f"SLAM: {self.slam.tracking_status} | Map: {len(self.slam.map_points)} pts"
            # Emerald green if tracking is OK, red/coral if tracking lost
            slam_color = (34, 197, 94) if "OK" in self.slam.tracking_status else (68, 68, 239)
        else:
            # Android status indicates if relative calibration is successful or pending
            status_str = "FUSED OK" if (fusion_manager is not None and fusion_manager.calibrated) else "CALIBRATING"
            slam_status = f"SLAM: {status_str}"
            slam_color = (34, 197, 94) if "FUSED" in status_str else (68, 180, 239) # Green if fused, amber if calibrating

        # Draw status label
        cv2.putText(annotated_frame, slam_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 23, 42), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, slam_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, slam_color, 1, cv2.LINE_AA)

        # Track which hand labels are active in this frame
        detected_labels = set()
        h, w, _ = frame.shape

        # Check if any hands were detected
        if results.multi_hand_landmarks and results.multi_handedness:
            # Loop through detected hand landmarks and their classifications
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Retrieve handedness details
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                detected_labels.add(hand_label)
                confidence = handedness.classification[0].score  # Detection confidence (float)
                
                # Feed to air-drawing system with depth map, SLAM engine, and fusion configurations
                self.drawer.process_hand(
                    hand_landmarks=hand_landmarks,
                    hand_label=hand_label,
                    frame_width=w,
                    frame_height=h,
                    depth_map=depth_map,
                    slam=self.slam,
                    camera_name=camera_name,
                    confidence=confidence,
                    fusion_manager=fusion_manager
                )
                
                # Draw standard hand links and joint dots
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_spec,
                    self.connection_spec
                )
                
                # Estimate 3D coordinates (physical camera coordinates in cm) for all 21 landmarks
                landmarks_3d = []
                for lm_id, lm in enumerate(hand_landmarks.landmark):
                    # Convert normalized relative coordinates to absolute pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cx = max(0, min(w - 1, cx))
                    cy = max(0, min(h - 1, cy))
                    
                    # Query disparity from depth map
                    if depth_map is not None:
                        d_val = depth_map[cy, cx]
                    else:
                        d_val = 128.0
                        
                    # Convert relative disparity [0, 255] to physical depth Z in cm (30cm to 120cm)
                    d_norm = d_val / 255.0
                    z_c = 30.0 + (1.0 - d_norm) * 90.0
                    
                    # Back-project using camera pinhole equations (focal length = w)
                    x_c = (cx - w / 2.0) * (z_c / w)
                    y_c = (cy - h / 2.0) * (z_c / w)
                    landmarks_3d.append((x_c, y_c, z_c))
                    
                    # Draw landmark ID text next to their coordinates
                    cv2.putText(
                        annotated_frame,
                        str(lm_id),
                        (cx + 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,              # Font scale
                        (248, 250, 252),  # White text (slate-50)
                        1,                 # Line thickness
                        cv2.LINE_AA
                    )
                
                # Get the 3D position of the drawing index finger tip (landmark 8)
                tip_3d = landmarks_3d[8]
                # Represent coordinates in fused world space if calibrated
                coord_frame = "W"
                if camera_name == "Android" and fusion_manager is not None and fusion_manager.calibrated:
                    tip_3d_webcam = fusion_manager.transform_android_to_webcam(tip_3d)
                    if fusion_manager.webcam_slam is not None and fusion_manager.webcam_slam.tracking_status == "Tracking OK":
                        tip_3d = fusion_manager.webcam_slam.transform_cam_to_world(tip_3d_webcam)
                elif camera_name == "Webcam" and self.slam.tracking_status == "Tracking OK":
                    tip_3d = self.slam.transform_cam_to_world(tip_3d)
                else:
                    coord_frame = "C"
                
                # Display classification label, confidence, and real-time XYZ coordinates near wrist (ID 0)
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                
                # Render label with dynamic color in BGR format (Green/Emerald for Right, Orange for Left)
                label_color = (129, 185, 16) if hand_label == "Right" else (11, 158, 245)
                info_text = f"{hand_label} ({confidence * 100:.0f}%) | {coord_frame}_X:{tip_3d[0]:.1f} Y:{tip_3d[1]:.1f} Z:{tip_3d[2]:.1f}"
                
                # Background text outline for high visibility against varying frame backdrops
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (wrist_x - 30, wrist_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (15, 23, 42),       # Dark shadow border
                    3,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (wrist_x - 30, wrist_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    label_color,        # Main text
                    1,
                    cv2.LINE_AA
                )
                
                # Append metadata
                hands_info.append({
                    "label": hand_label,
                    "confidence": confidence,
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                    "landmarks_3d": landmarks_3d
                })

        # Finalize drawing for hands that went out of frame
        for label in ["Left", "Right"]:
            if label not in detected_labels:
                self.drawer.finalize_hand(label)
                
        # Draw all fused paths using the primary coordinates projection SLAM reference
        render_slam = self.slam if camera_name == "Webcam" else (fusion_manager.webcam_slam if fusion_manager is not None else None)
        self.drawer.draw_on_frame(annotated_frame, render_slam)

        # Stream SLAM and drawing data to Unity 3D Engine in real time if connection is open
        if camera_name == "Webcam" or (fusion_manager is not None and not fusion_manager.calibrated):
            # Stream using main camera parameters
            self.unity_bridge.stream_frame_data(
                cam_pos=self.slam.trajectory[-1] if camera_name == "Webcam" else [0, 0, 0],
                cam_rot_3x3=self.slam.R_w if camera_name == "Webcam" else np.identity(3),
                point_cloud_pts=self.slam.map_points if camera_name == "Webcam" else [],
                drawing_strokes=self.drawer.strokes + list(self.drawer.active_strokes.values()),
                shapes_list=self.drawer.recognized_shapes
            )
        elif camera_name == "Android" and fusion_manager is not None and fusion_manager.calibrated:
            # Stream Android transformed camera parameters
            android_center_webcam = fusion_manager.transform_android_to_webcam((0, 0, 0))
            if fusion_manager.webcam_slam is not None:
                android_center_world = fusion_manager.webcam_slam.transform_cam_to_world(android_center_webcam)
                # Compute Android camera rotation in world: R_android_world = R_webcam_world * R_android_to_webcam
                rot_world = np.dot(fusion_manager.webcam_slam.R_w, fusion_manager.R_android_to_webcam)
            else:
                android_center_world = android_center_webcam
                rot_world = fusion_manager.R_android_to_webcam
                
            self.unity_bridge.stream_frame_data(
                cam_pos=android_center_world,
                cam_rot_3x3=rot_world,
                point_cloud_pts=fusion_manager.webcam_slam.map_points if fusion_manager.webcam_slam is not None else [],
                drawing_strokes=self.drawer.strokes + list(self.drawer.active_strokes.values()),
                shapes_list=self.drawer.recognized_shapes
            )

        return annotated_frame, hands_info

    def clear_drawing(self):
        """Clears all drawn paths on this detector's canvas and resets SLAM pose."""
        self.drawer.clear()
        self.slam.reset()

    def set_drawing_color(self, color_bgr):
        """Sets the drawing stroke color in BGR format."""
        self.drawer.color = color_bgr

    def set_drawing_thickness(self, thickness):
        """Sets the drawing stroke thickness in pixels."""
        self.drawer.thickness = thickness

    def save_session(self, filepath, fusion_manager=None):
        """Saves current SLAM trajectory, map points, drawing strokes, shapes, colors, and properties to a JSON file."""
        import json
        try:
            data = {
                "strokes": [ [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in stroke] for stroke in self.drawer.strokes ],
                "trajectory": [ [float(pt[0]), float(pt[1]), float(pt[2])] for pt in self.slam.trajectory ],
                "map_points": [ [float(pt[0]), float(pt[1]), float(pt[2])] for pt in self.slam.map_points ],
                "R_w": self.slam.R_w.tolist(),
                "t_w": self.slam.t_w.tolist(),
                "recognized_shapes": self.drawer.recognized_shapes,
                "drawing_color": list(self.drawer.color),
                "drawing_thickness": int(self.drawer.thickness)
            }
            if fusion_manager is not None and fusion_manager.calibrated:
                data["fusion_calibration"] = {
                    "R": fusion_manager.R_android_to_webcam.tolist(),
                    "t": fusion_manager.t_android_to_webcam.tolist()
                }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            return False
            
    def load_session(self, filepath, fusion_manager=None):
        """Loads SLAM trajectory, map points, drawing strokes, shapes, colors, and calibration from a JSON file."""
        import json
        import os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            self.drawer.strokes = [ [tuple(pt) for pt in stroke] for stroke in data.get("strokes", []) ]
            self.slam.trajectory = [ np.array(pt, dtype=np.float32) for pt in data.get("trajectory", [np.zeros(3, dtype=np.float32)]) ]
            self.slam.map_points = [ np.array(pt, dtype=np.float32) for pt in data.get("map_points", []) ]
            self.slam.R_w = np.array(data.get("R_w", np.identity(3).tolist()), dtype=np.float32)
            self.slam.t_w = np.array(data.get("t_w", np.zeros((3, 1)).tolist()), dtype=np.float32)
            self.drawer.recognized_shapes = data.get("recognized_shapes", [])
            
            color_lst = data.get("drawing_color", [229, 70, 79])
            self.drawer.color = (color_lst[0], color_lst[1], color_lst[2])
            self.drawer.thickness = data.get("drawing_thickness", 4)
            
            if fusion_manager is not None and "fusion_calibration" in data:
                calib = data["fusion_calibration"]
                fusion_manager.R_android_to_webcam = np.array(calib["R"], dtype=np.float32)
                fusion_manager.t_android_to_webcam = np.array(calib["t"], dtype=np.float32)
                fusion_manager.calibrated = True
                
            self.slam.tracking_status = "Tracking OK"
            return True
        except Exception as e:
            return False
