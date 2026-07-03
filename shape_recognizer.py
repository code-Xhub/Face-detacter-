import numpy as np
import math
import cv2

class ShapeRecognizer:
    """
    Real-time shape recognition engine implementing a modified 2D $1 Gesture Recognizer 
    coupled with geometric heuristics (aspect ratios, path ratios, and angular winding)
    to classify hand-drawn strokes.
    
    Recognizes:
    - Circle
    - Square
    - Rectangle
    - Triangle
    - Star
    - Arrow
    - Line
    - Heart
    - Spiral
    """
    def __init__(self, num_points=64, square_size=250.0):
        self.num_points = num_points
        self.square_size = square_size
        self.templates = {}
        self.generate_templates()

    def generate_templates(self):
        """Generates mathematical templates for all required shapes and normalizes them."""
        raw_templates = {
            "Circle": self._create_circle(),
            "Square": self._create_square(),
            "Rectangle": self._create_rectangle(),
            "Triangle": self._create_triangle(),
            "Star": self._create_star(),
            "Arrow": self._create_arrow(),
            "Line": self._create_line(),
            "Heart": self._create_heart(),
            "Spiral": self._create_spiral()
        }
        
        # Normalize all templates to ensure alignment during comparison
        for name, pts in raw_templates.items():
            norm_pts, _, _ = self.normalize(pts)
            self.templates[name] = norm_pts

    # --- Programmatic Template Generators ---
    
    def _create_circle(self):
        angles = np.linspace(0, 2 * np.pi, self.num_points)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def _create_square(self):
        pts = []
        pts_side = self.num_points // 4
        # Side 1: Bottom
        for x in np.linspace(-1, 1, pts_side, endpoint=False):
            pts.append([x, -1])
        # Side 2: Right
        for y in np.linspace(-1, 1, pts_side, endpoint=False):
            pts.append([1, y])
        # Side 3: Top
        for x in np.linspace(1, -1, pts_side, endpoint=False):
            pts.append([x, 1])
        # Side 4: Left
        for y in np.linspace(1, -1, self.num_points - len(pts)):
            pts.append([-1, y])
        return np.array(pts, dtype=np.float32)

    def _create_rectangle(self):
        pts = []
        # Width = 2, Height = 1 (Perimeter = 6)
        n1 = int(self.num_points * 2 / 6)
        n2 = int(self.num_points * 1 / 6)
        n3 = int(self.num_points * 2 / 6)
        n4 = self.num_points - n1 - n2 - n3
        
        for x in np.linspace(-1, 1, n1, endpoint=False):
            pts.append([x, -0.5])
        for y in np.linspace(-0.5, 0.5, n2, endpoint=False):
            pts.append([1, y])
        for x in np.linspace(1, -1, n3, endpoint=False):
            pts.append([x, 0.5])
        for y in np.linspace(0.5, -0.5, n4):
            pts.append([-1, y])
        return np.array(pts, dtype=np.float32)

    def _create_triangle(self):
        pts = []
        pts_side = self.num_points // 3
        # Equilateral Triangle corners
        v0 = [0.0, 1.0]
        v1 = [0.866, -0.5]
        v2 = [-0.866, -0.5]
        
        for t in np.linspace(0, 1, pts_side, endpoint=False):
            pts.append([(1 - t) * v0[0] + t * v1[0], (1 - t) * v0[1] + t * v1[1]])
        for t in np.linspace(0, 1, pts_side, endpoint=False):
            pts.append([(1 - t) * v1[0] + t * v2[0], (1 - t) * v1[1] + t * v2[1]])
        for t in np.linspace(0, 1, self.num_points - len(pts)):
            pts.append([(1 - t) * v2[0] + t * v0[0], (1 - t) * v2[1] + t * v0[1]])
        return np.array(pts, dtype=np.float32)

    def _create_star(self):
        vertices = []
        for i in range(10):
            angle = i * np.pi / 5.0 - np.pi / 2.0
            r = 1.0 if i % 2 == 0 else 0.38
            vertices.append([r * np.cos(angle), r * np.sin(angle)])
        vertices.append(vertices[0])
        
        pts = []
        pts_seg = self.num_points // 10
        for i in range(10):
            v1 = vertices[i]
            v2 = vertices[i+1]
            count = pts_seg if i < 9 else (self.num_points - len(pts))
            for t in np.linspace(0, 1, count, endpoint=(i==9)):
                pts.append([(1 - t) * v1[0] + t * v2[0], (1 - t) * v1[1] + t * v2[1]])
        return np.array(pts, dtype=np.float32)

    def _create_arrow(self):
        # Single stroke arrow: Shaft, then upper head, then back, then lower head
        n1 = int(self.num_points * 0.45)
        n2 = int(self.num_points * 0.18)
        n3 = int(self.num_points * 0.18)
        n4 = self.num_points - n1 - n2 - n3
        
        pts = []
        for x in np.linspace(-1, 1, n1, endpoint=False):
            pts.append([x, 0.0])
        for t in np.linspace(0, 1, n2, endpoint=False):
            pts.append([(1 - t) * 1.0 + t * 0.5, (1 - t) * 0.0 + t * 0.5])
        for t in np.linspace(0, 1, n3, endpoint=False):
            pts.append([(1 - t) * 0.5 + t * 1.0, (1 - t) * 0.5 + t * 0.0])
        for t in np.linspace(0, 1, n4):
            pts.append([(1 - t) * 1.0 + t * 0.5, (1 - t) * 0.0 + t * -0.5])
        return np.array(pts, dtype=np.float32)

    def _create_line(self):
        pts = []
        for x in np.linspace(-1, 1, self.num_points):
            pts.append([x, 0.0])
        return np.array(pts, dtype=np.float32)

    def _create_heart(self):
        t = np.linspace(0, 2 * np.pi, self.num_points)
        x = 16 * (np.sin(t) ** 3)
        # Flip Y to keep heart right-side up in image coordinate space (where Y goes down)
        y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        pts = np.stack([x, y], axis=1)
        # Normalize range to [-1, 1]
        min_val = np.min(pts, axis=0)
        max_val = np.max(pts, axis=0)
        pts = (pts - min_val) / (max_val - min_val) * 2.0 - 1.0
        return pts

    def _create_spiral(self):
        # Archimedean spiral
        theta = np.linspace(0.1, 3.5 * np.pi, self.num_points)
        x = theta * np.cos(theta)
        y = theta * np.sin(theta)
        pts = np.stack([x, y], axis=1)
        min_val = np.min(pts, axis=0)
        max_val = np.max(pts, axis=0)
        pts = (pts - min_val) / (max_val - min_val) * 2.0 - 1.0
        return pts

    # --- Preprocessing & Normalization ---

    def resample(self, points, n):
        """Resamples points so that they are equidistant along the path length."""
        if len(points) == 0:
            return []
        if len(points) == 1:
            return np.repeat(points, n, axis=0)

        pts = np.array(points, dtype=np.float32)
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        path_length = np.sum(dists)
        
        if path_length < 1e-5:
            # Handle degenerate path (all points same or very close)
            return np.repeat(pts[:1], n, axis=0)
            
        interval = path_length / (n - 1)
        resampled = [pts[0]]
        accum_dist = 0.0
        
        i = 1
        curr_pt = pts[0]
        while i < len(pts):
            d = np.linalg.norm(pts[i] - curr_pt)
            if accum_dist + d >= interval:
                t = (interval - accum_dist) / d
                new_pt = curr_pt + t * (pts[i] - curr_pt)
                resampled.append(new_pt)
                accum_dist = 0.0
                curr_pt = new_pt
            else:
                accum_dist += d
                curr_pt = pts[i]
                i += 1
                
        while len(resampled) < n:
            resampled.append(pts[-1])
        if len(resampled) > n:
            resampled = resampled[:n]
            
        return np.array(resampled, dtype=np.float32)

    def rotate_by(self, points, angle, centroid):
        """Rotates points around a centroid by the specified angle."""
        c = np.cos(angle)
        s = np.sin(angle)
        
        dx = points[:, 0] - centroid[0]
        dy = points[:, 1] - centroid[1]
        
        rx = dx * c - dy * s + centroid[0]
        ry = dx * s + dy * c + centroid[1]
        
        return np.stack([rx, ry], axis=1)

    def scale_to(self, points, size):
        """Scales points uniformly to fit a bounding box of size x size."""
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        
        # Use uniform scaling to preserve aspect ratio (crucial for squares vs rectangles & lines)
        scale = size / max(w, h, 1e-5)
        scaled_pts = points * scale
        return scaled_pts, w, h

    def translate_to(self, points, origin=(0.0, 0.0)):
        """Translates points so that their centroid is at origin."""
        centroid = np.mean(points, axis=0)
        return points - centroid + np.array(origin)

    def normalize(self, points):
        """Applies full preprocessing pipeline to raw points."""
        # 1. Resample to standard point count
        pts = self.resample(points, self.num_points)
        # 2. Find centroid and rotate to align first point to angle 0
        centroid = np.mean(pts, axis=0)
        theta = np.arctan2(pts[0][1] - centroid[1], pts[0][0] - centroid[0])
        pts = self.rotate_by(pts, -theta, centroid)
        # 3. Uniformly scale
        pts, w, h = self.scale_to(pts, self.square_size)
        # 4. Translate centroid to origin (0, 0)
        pts = self.translate_to(pts, (0.0, 0.0))
        return pts, w, h

    # --- Distance Metrics ---

    def path_distance(self, pts1, pts2):
        """Calculates average Euclidean distance between corresponding points."""
        return np.mean(np.linalg.norm(pts1 - pts2, axis=1))

    def distance_at_best_angle(self, pts, template, a_min=-np.pi/4, a_max=np.pi/4, a_step=math.radians(2)):
        """Finds minimum distance between candidate and template searching over a range of angles and directions."""
        min_dist = float('inf')
        centroid = np.mean(pts, axis=0)
        
        # Test both drawing directions: forward and backward
        for test_pts in [pts, np.flip(pts, axis=0)]:
            for angle in np.arange(a_min, a_max + a_step, a_step):
                rotated = self.rotate_by(test_pts, angle, centroid)
                aligned = rotated - np.mean(rotated, axis=0) # ensure origin alignment
                dist = self.path_distance(aligned, template)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    # --- Shape Classification ---

    def classify(self, raw_points):
        """
        Classifies a raw 2D stroke into one of the 9 shapes.
        Returns:
            tuple: (shape_name, confidence_percentage)
        """
        if len(raw_points) < 8:
            return "Unknown Shape", 0.0

        pts_arr = np.array(raw_points, dtype=np.float32)
        
        # Calculate raw stroke characteristics for geometric heuristics
        # 1. Path length vs start-to-end straight line distance (Line check)
        dists = np.linalg.norm(pts_arr[1:] - pts_arr[:-1], axis=1)
        total_length = np.sum(dists)
        start_end_dist = np.linalg.norm(pts_arr[-1] - pts_arr[0])
        line_ratio = start_end_dist / total_length if total_length > 0 else 0.0
        
        # If the stroke is almost perfectly straight, classify as Line immediately
        if line_ratio > 0.94:
            # We scale the confidence between 75% and 99% based on straightness
            conf = 75.0 + (line_ratio - 0.94) / (1.0 - 0.94) * 24.0
            return "Line", conf

        # Normalize the candidate gesture
        norm_pts, raw_w, raw_h = self.normalize(pts_arr)
        
        # Calculate raw aspect ratio (important to distinguish Square from Rectangle)
        raw_aspect_ratio = max(raw_w, raw_h) / max(min(raw_w, raw_h), 1e-5)

        # 2. Spiral Check via cumulative winding angle around centroid
        centroid = np.mean(pts_arr, axis=0)
        vectors = pts_arr - centroid
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        # Unwrapped angles to calculate total continuous winding
        unwrapped_angles = np.unwrap(angles)
        total_angle_change = abs(unwrapped_angles[-1] - unwrapped_angles[0])
        
        # Spiral winding heuristic: spirals wind at least 420 degrees (approx 2.33 * pi)
        if total_angle_change > 2.33 * np.pi:
            # Calculate distance correlation to verify inwards/outwards spiraling pattern
            pt_dists = np.linalg.norm(vectors, axis=1)
            indices = np.arange(len(pt_dists))
            correlation = np.corrcoef(indices, pt_dists)[0, 1]
            
            # If radius decreases/increases with time (high correlation), it's a Spiral
            if abs(correlation) > 0.65:
                conf = 70.0 + min(30.0, (total_angle_change - 2.33 * np.pi) * 10.0)
                return "Spiral", conf

        # 3. Match against remaining templates
        best_shape = "Unknown Shape"
        best_dist = float('inf')
        
        for name, template in self.templates.items():
            # Skip Spiral and Line template matching if heuristics didn't trigger, or test them anyway
            dist = self.distance_at_best_angle(norm_pts, template)
            if dist < best_dist:
                best_dist = dist
                best_shape = name

        # Map distance to confidence percentage: 0 distance -> 100%, 110 distance -> 0%
        # Threshold 70% corresponds to distance <= 33.0 pixels
        confidence = max(0.0, 1.0 - best_dist / 110.0) * 100.0

        # 4. Apply refinement overrides
        
        # Square vs Rectangle refinement based on aspect ratio
        if best_shape in ["Square", "Rectangle"]:
            if raw_aspect_ratio < 1.18:
                best_shape = "Square"
            else:
                best_shape = "Rectangle"
                
        # Spiral override: if it matches spiral template but didn't trigger the winding check, 
        # make sure it winds at least 1.5 * pi to be valid, otherwise fall back to Circle/Unknown
        if best_shape == "Spiral" and total_angle_change < 1.5 * np.pi:
            # Re-evaluate next best match
            best_dist = float('inf')
            best_shape = "Unknown Shape"
            for name, template in self.templates.items():
                if name == "Spiral":
                    continue
                dist = self.distance_at_best_angle(norm_pts, template)
                if dist < best_dist:
                    best_dist = dist
                    best_shape = name
            confidence = max(0.0, 1.0 - best_dist / 110.0) * 100.0
            
        # Circle vs other shapes: circles should have low distance variance
        if best_shape == "Circle":
            dists_from_center = np.linalg.norm(norm_pts, axis=1)
            var_dist = np.std(dists_from_center) / np.mean(dists_from_center)
            # If variance is too high, it's not a clean circle
            if var_dist > 0.22:
                confidence *= 0.8  # penalize circle confidence

        # Limit maximum confidence to 99.5% for visual authenticity unless perfect
        if confidence > 99.5:
            confidence = 99.5
            
        return best_shape, confidence

    def generate_perfect_shape_2d(self, shape_name, pts_2d):
        """
        Generates 2D points of a perfect geometric outline centered, scaled,
        and rotated to align with the given user drawing points.
        """
        if len(pts_2d) < 3:
            return pts_2d

        # 1. Line
        if shape_name == "Line":
            p0 = pts_2d[0]
            p1 = pts_2d[-1]
            pts = []
            for t in np.linspace(0, 1, self.num_points):
                pts.append((1.0 - t) * p0 + t * p1)
            return np.array(pts, dtype=np.float32)

        # 2. Circle
        elif shape_name == "Circle":
            c = np.mean(pts_2d, axis=0)
            r = np.mean(np.linalg.norm(pts_2d - c, axis=1))
            angles = np.linspace(0, 2 * np.pi, self.num_points)
            return np.stack([c[0] + r * np.cos(angles), c[1] + r * np.sin(angles)], axis=1)

        # 3. Square or Rectangle
        elif shape_name in ["Square", "Rectangle"]:
            rect = cv2.minAreaRect(pts_2d)
            (cx, cy), (w, h), angle = rect
            if shape_name == "Square":
                s = (w + h) / 2.0
                w, h = s, s
            box = cv2.boxPoints(((cx, cy), (w, h), angle))
            
            pts = []
            pts_side = self.num_points // 4
            for i in range(4):
                v1 = box[i]
                v2 = box[(i + 1) % 4]
                for t in np.linspace(0, 1, pts_side, endpoint=False):
                    pts.append((1.0 - t) * v1 + t * v2)
            pts.append(box[0]) # close loop
            return np.array(pts, dtype=np.float32)

        # 4. Triangle
        elif shape_name == "Triangle":
            rect = cv2.minAreaRect(pts_2d)
            (cx, cy), (w, h), angle = rect
            
            # Triangle corners in local space relative to bounding box
            v = np.array([
                [0.0, -h / 2.0],
                [w / 2.0, h / 2.0],
                [-w / 2.0, h / 2.0]
            ], dtype=np.float32)
            
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rot_v = []
            for pt in v:
                rx = pt[0] * cos_a - pt[1] * sin_a + cx
                ry = pt[0] * sin_a + pt[1] * cos_a + cy
                rot_v.append([rx, ry])
            rot_v = np.array(rot_v, dtype=np.float32)
            
            pts = []
            pts_side = self.num_points // 3
            for i in range(3):
                v1 = rot_v[i]
                v2 = rot_v[(i + 1) % 3]
                for t in np.linspace(0, 1, pts_side, endpoint=False):
                    pts.append((1.0 - t) * v1 + t * v2)
            pts.append(rot_v[0])
            return np.array(pts, dtype=np.float32)

        # 5. Star
        elif shape_name == "Star":
            rect = cv2.minAreaRect(pts_2d)
            (cx, cy), (w, h), angle = rect
            
            r_out = max(w, h) / 2.0
            r_in = r_out * 0.38
            
            vertices = []
            for i in range(10):
                ang = i * np.pi / 5.0 - np.pi / 2.0
                r = r_out if i % 2 == 0 else r_in
                vertices.append([r * np.cos(ang), r * np.sin(ang)])
            vertices.append(vertices[0])
            
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rot_v = []
            for pt in vertices:
                rx = pt[0] * cos_a - pt[1] * sin_a + cx
                ry = pt[0] * sin_a + pt[1] * cos_a + cy
                rot_v.append([rx, ry])
            rot_v = np.array(rot_v, dtype=np.float32)
            
            pts = []
            pts_seg = self.num_points // 10
            for i in range(10):
                v1 = rot_v[i]
                v2 = rot_v[i + 1]
                count = pts_seg if i < 9 else (self.num_points - len(pts))
                for t in np.linspace(0, 1, count, endpoint=(i == 9)):
                    pts.append((1.0 - t) * v1 + t * v2)
            return np.array(pts, dtype=np.float32)

        # 6. Arrow
        elif shape_name == "Arrow":
            p0 = pts_2d[0]
            p1 = pts_2d[-1]
            v = p1 - p0
            L = np.linalg.norm(v)
            if L < 1e-5:
                return pts_2d
            u = v / L
            L_wing = L * 0.25
            
            ang1 = math.radians(150)
            u1 = np.array([u[0] * math.cos(ang1) - u[1] * math.sin(ang1), u[0] * math.sin(ang1) + u[1] * math.cos(ang1)], dtype=np.float32)
            ang2 = math.radians(-150)
            u2 = np.array([u[0] * math.cos(ang2) - u[1] * math.sin(ang2), u[0] * math.sin(ang2) + u[1] * math.cos(ang2)], dtype=np.float32)
            
            pts = []
            n_shaft = int(self.num_points * 0.45)
            n_wing = int(self.num_points * 0.18)
            n_back = int(self.num_points * 0.18)
            n_last = self.num_points - n_shaft - n_wing - n_back
            
            # Shaft: Tail to Tip
            for t in np.linspace(0, 1, n_shaft, endpoint=False):
                pts.append((1.0 - t) * p0 + t * p1)
            # Tip to Wing1
            w1 = p1 + u1 * L_wing
            for t in np.linspace(0, 1, n_wing, endpoint=False):
                pts.append((1.0 - t) * p1 + t * w1)
            # Wing1 to Tip
            for t in np.linspace(0, 1, n_back, endpoint=False):
                pts.append((1.0 - t) * w1 + t * p1)
            # Tip to Wing2
            w2 = p1 + u2 * L_wing
            for t in np.linspace(0, 1, n_last):
                pts.append((1.0 - t) * p1 + t * w2)
            return np.array(pts, dtype=np.float32)

        # 7. Heart
        elif shape_name == "Heart":
            rect = cv2.minAreaRect(pts_2d)
            (cx, cy), (w, h), angle = rect
            
            t = np.linspace(0, 2 * np.pi, self.num_points)
            hx = np.sin(t) ** 3
            hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16.0
            
            hx = hx * (w / 2.0)
            hy = hy * (h / 2.0)
            
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            pts = []
            for i in range(self.num_points):
                rx = hx[i] * cos_a - hy[i] * sin_a + cx
                ry = hx[i] * sin_a + hy[i] * cos_a + cy
                pts.append([rx, ry])
            return np.array(pts, dtype=np.float32)

        # 8. Spiral
        elif shape_name == "Spiral":
            c = np.mean(pts_2d, axis=0)
            R = np.max(np.linalg.norm(pts_2d - c, axis=1))
            if R < 1e-5:
                return pts_2d
                
            theta_max = 3.5 * np.pi
            t = np.linspace(0.1, theta_max, self.num_points)
            r = (t / theta_max) * R
            
            pts = []
            for i in range(self.num_points):
                rx = c[0] + r[i] * np.cos(t[i])
                ry = c[1] + r[i] * np.sin(t[i])
                pts.append([rx, ry])
            return np.array(pts, dtype=np.float32)

        return pts_2d
