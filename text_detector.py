"""
DBNet Text Detector - Lightweight ONNX-based text detection
Uses PaddleOCR's en_PP-OCRv3_det model (2.4 MB) via onnxruntime-gpu.
Detects text regions and returns bounding boxes for black-box masking.
"""
import cv2
import numpy as np
import os

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("Warning: onnxruntime not installed. Text detection disabled.")


class DBNetTextDetector:
    """
    DBNet text detector using ONNX Runtime with GPU support.
    Runs inference every N frames and caches results for performance.
    """

    def __init__(self, model_path="models/dbnet_en.onnx", use_gpu=True):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.ready = False

        # Detection parameters
        self.conf_threshold = 0.2       # Binary threshold for probability map
        self.box_threshold = 0.3        # Min score to keep a box
        self.min_area = 50              # Min pixel area to keep a text region
        self.max_side = 640             # Max input dimension (speed vs accuracy)
        self.unclip_ratio = 1.6         # Expand detected polygons slightly

        # Caching
        self.cached_text_boxes = []
        self.text_frame_count = 0
        self.text_detection_interval = 8  # Run every N frames (text is static)
        self.text_persistence = 0
        self.max_text_persistence = 20    # Keep text boxes for 20 frames

        if not HAS_ORT:
            print("onnxruntime not available – text detection disabled")
            return

        if not os.path.exists(model_path):
            print(f"DBNet model not found at {model_path}")
            return

        self._load_model(model_path, use_gpu)

    def _load_model(self, model_path, use_gpu):
        """Load ONNX model with GPU or CPU provider."""
        try:
            providers = []
            if use_gpu:
                # Try CUDA first, then DirectML, then CPU
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if "DmlExecutionProvider" in available:
                    providers.append("DmlExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Keep it lightweight – single thread for text detection
            sess_options.intra_op_num_threads = 2

            self.session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.ready = True

            active_provider = self.session.get_providers()[0]
            print(f"DBNet text detector loaded on {active_provider}  "
                  f"(model: {os.path.basename(model_path)}, "
                  f"{os.path.getsize(model_path) / 1e6:.1f} MB)")

        except Exception as e:
            print(f"Error loading DBNet model: {e}")
            self.session = None
            self.ready = False

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #
    def _preprocess(self, image):
        """
        Prepare frame for DBNet inference (PaddleOCR format).
        - Input is BGR (from OpenCV) — do NOT convert to RGB
        - Resize so longest side <= max_side, both dims multiple of 32
        - Normalize with PaddleOCR mean/std (applied to BGR channels)
        - Convert HWC→NCHW float32
        Returns (blob, ratio_h, ratio_w)
        """
        orig_h, orig_w = image.shape[:2]

        # Scale so longest side <= max_side
        ratio = 1.0
        if max(orig_h, orig_w) > self.max_side:
            ratio = self.max_side / max(orig_h, orig_w)

        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)

        # Round to multiples of 32
        new_h = max(32, (new_h + 31) // 32 * 32)
        new_w = max(32, (new_w + 31) // 32 * 32)

        resized = cv2.resize(image, (new_w, new_h))

        # Normalize (PaddleOCR mean/std, applied directly to BGR channels)
        img = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # HWC → NCHW
        blob = img.transpose(2, 0, 1)[np.newaxis, ...]

        ratio_h = orig_h / new_h
        ratio_w = orig_w / new_w

        return blob.astype(np.float32), ratio_h, ratio_w

    # ------------------------------------------------------------------ #
    #  Postprocessing
    # ------------------------------------------------------------------ #
    def _postprocess(self, prob_map, ratio_h, ratio_w, orig_h, orig_w):
        """
        Convert probability map → list of (x, y, w, h) bounding boxes.
        """
        # prob_map shape: (1, 1, H, W)
        pred = prob_map[0, 0]

        # Binary threshold
        binary = (pred > self.conf_threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Score: mean probability inside the contour
            mask = np.zeros_like(pred, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            score = cv2.mean(pred, mask)[0]
            if score < self.box_threshold:
                continue

            # Get bounding rect and unclip (expand slightly)
            rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rect)

            # Unclip: expand the polygon
            box_points = self._unclip(box_points, self.unclip_ratio)

            # Get axis-aligned bounding box
            x_coords = box_points[:, 0] * ratio_w
            y_coords = box_points[:, 1] * ratio_h

            x_min = max(0, int(np.min(x_coords)))
            y_min = max(0, int(np.min(y_coords)))
            x_max = min(orig_w, int(np.max(x_coords)))
            y_max = min(orig_h, int(np.max(y_coords)))

            w = x_max - x_min
            h = y_max - y_min

            if w > 5 and h > 3:  # Filter tiny noise
                boxes.append((x_min, y_min, w, h))

        return boxes

    def _unclip(self, box_points, ratio):
        """Expand a rotated rect's box points by ratio (Vatti clipping approx)."""
        # Simple expansion: scale from center
        cx = np.mean(box_points[:, 0])
        cy = np.mean(box_points[:, 1])
        expanded = box_points.copy()
        expanded[:, 0] = cx + (box_points[:, 0] - cx) * ratio
        expanded[:, 1] = cy + (box_points[:, 1] - cy) * ratio
        return expanded

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame):
        """
        Run DBNet text detection on a frame.
        Returns list of (x, y, w, h) bounding boxes for text regions.
        """
        if not self.ready:
            return []

        try:
            orig_h, orig_w = frame.shape[:2]

            # Use BGR directly — PaddleOCR models expect BGR input
            blob, ratio_h, ratio_w = self._preprocess(frame)

            # Run inference
            prob_map = self.session.run(
                [self.output_name], {self.input_name: blob}
            )[0]

            boxes = self._postprocess(prob_map, ratio_h, ratio_w, orig_h, orig_w)

            return boxes

        except Exception as e:
            print(f"DBNet detection error: {e}")
            return []

    def detect_cached(self, frame):
        """
        Detect text with frame-skipping cache for performance.
        Only runs inference every N frames; returns cached boxes otherwise.
        """
        self.text_frame_count += 1

        if self.text_frame_count >= self.text_detection_interval:
            self.text_frame_count = 0
            new_boxes = self.detect(frame)

            if len(new_boxes) > 0:
                self.cached_text_boxes = new_boxes
                self.text_persistence = self.max_text_persistence
            elif self.text_persistence > 0:
                self.text_persistence -= 1
            else:
                self.cached_text_boxes = []

        return self.cached_text_boxes


def apply_text_blackbox(frame, bbox, padding=5):
    """
    Draw a plain black rectangle over a text region.
    Simple, clean, and completely hides the text.

    Args:
        frame: Input image (numpy array, BGR)
        bbox: Tuple (x, y, w, h) defining the text region
        padding: Extra pixels around the text box

    Returns:
        Modified frame with black box over text
    """
    x, y, w, h = bbox
    img_h, img_w = frame.shape[:2]

    x0 = max(0, int(x - padding))
    y0 = max(0, int(y - padding))
    x1 = min(img_w, int(x + w + padding))
    y1 = min(img_h, int(y + h + padding))

    # Plain black rectangle — fully opaque
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    return frame
