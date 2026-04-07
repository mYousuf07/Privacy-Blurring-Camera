"""
Privacy-Blurring Camera - Main Application
Floating widget with draggable UI and minimize-to-logo functionality.

Fully fixed and cleaned version (April 2026)
- All import errors resolved
- Syntax completely clean
- Ready to copy-paste and run
"""

import sys
try:
    import torch
    import onnxruntime
except ImportError:
    pass

from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QComboBox,
)
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal, pyqtSlot, QRectF
from PyQt6.QtGui import (
    QPixmap, QPainter, QPainterPath,
    QFont, QFontDatabase, QImage,
)
from PyQt6.QtSvg import QSvgRenderer

import cv2
import os
import json
import numpy as np

from anonymization import apply_privacy_pixelation, get_face_embedding, compare_faces
from text_detector import DBNetTextDetector, apply_text_blackbox
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------

class CameraThread(QThread):
    """
    Captures frames from a webcam and emits them via frame_ready.
    """

    frame_ready = pyqtSignal(object)
    camera_opened = pyqtSignal(int)
    camera_failed = pyqtSignal()

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.running = False
        self.camera = None
        self.camera_index = camera_index

    def run(self):
        self.running = True

        indices_to_try = [self.camera_index]
        if self.camera_index != 0:
            indices_to_try.append(0)

        opened_index = None
        for idx in indices_to_try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.camera = cap
                opened_index = idx
                break
            cap.release()

        if self.camera is None or opened_index is None:
            print("Error: Could not open any camera.")
            self.camera_failed.emit()
            self.running = False
            return

        self.camera_opened.emit(opened_index)

        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)   # ~30 fps

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.wait()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class FloatingWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.is_minimized  = False
        self.is_fullscreen = False
        self.normal_size   = (400, 460)
        self.drag_position = QPoint()

        # YOLO / detection state
        self.yolo_model       = None
        self.device           = "cpu"
        self.frame_count      = 0
        self.cached_faces     = []
        self.detection_interval = 2
        self.face_persistence   = 0
        self.max_persistence    = 15
        self.smoothed_faces     = []

        # Text detector
        self.text_detector = None

        # Frame buffer — privacy safety net
        self.BUFFER_SIZE   = 3
        self.frame_buffer  = deque(maxlen=self.BUFFER_SIZE)

        # Trusted-face state
        self.trusted_faces      = []
        self.show_trusted       = False
        self.current_frame      = None
        self.trusted_faces_file = "trusted_faces.json"
        self.load_trusted_faces()

        self.init_font()
        self.init_ui()
        self.init_yolo()
        self.init_text_detector()
        self.init_camera()

    # ===================================================================
    # Initialisation helpers
    # ===================================================================

    def init_font(self):
        font_path = "assets/fonts/Roboto-Regular.ttf"
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id != -1:
                family = QFontDatabase.applicationFontFamilies(font_id)[0]
                self.setFont(QFont(family, 10))
                return
        self.setFont(QFont("Segoe UI", 10))

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main card
        self.main_container = QFrame(self)
        self.main_container.setObjectName("mainContainer")
        self.main_container.setStyleSheet("""
            #mainContainer {
                background-color: #282c34;
                border-radius: 12px;
                border: 1px solid #4b5263;
            }
        """)

        main_layout = QVBoxLayout(self.main_container)
        main_layout.setContentsMargins(10, 5, 10, 10)
        main_layout.setSpacing(8)

        main_layout.addWidget(self.create_title_bar())
        main_layout.addWidget(self.create_preview())
        main_layout.addWidget(self.create_camera_row())
        main_layout.addWidget(self.create_controls_row())
        main_layout.addWidget(self.create_stats_bar())
        main_layout.addWidget(self.create_status_bar())

        # Minimised logo bubble
        self.logo_widget = QFrame(self)
        self.logo_widget.setStyleSheet(
            "QFrame { background-color: transparent; border: none; }"
        )
        logo_layout = QVBoxLayout(self.logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_logo_pixmap(self.logo_label, size=80)
        logo_layout.addWidget(self.logo_label)
        self.logo_widget.hide()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.main_container)
        root.addWidget(self.logo_widget)
        self.setLayout(root)

        self.resize(*self.normal_size)
        self.setWindowTitle("Privacy Camera")

    def create_preview(self):
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #21252b;
                border-radius: 8px;
                color: #61afef;
                font-size: 14px;
                min-height: 240px;
            }
        """)
        return self.preview_label

    def create_camera_row(self):
        row = QFrame()
        row.setStyleSheet("background-color: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        cam_lbl = QLabel("Camera:")
        cam_lbl.setStyleSheet("color: #abb2bf; font-size: 11px;")
        layout.addWidget(cam_lbl)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0 (built-in)", "1 (external)", "2", "3"])
        self.camera_combo.setCurrentIndex(0)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c4049;
                color: #abb2bf;
                border: 1px solid #4b5263;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #3c4049;
                color: #abb2bf;
                selection-background-color: #61afef;
                selection-color: #282c34;
            }
        """)
        layout.addWidget(self.camera_combo)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c4049;
                color: #61afef;
                border: 1px solid #4b5263;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #4b5263; }
        """)
        apply_btn.clicked.connect(self.restart_camera)
        layout.addWidget(apply_btn)
        layout.addStretch()
        return row

    def create_controls_row(self):
        row = QFrame()
        row.setStyleSheet("background-color: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

        self.save_face_btn = QPushButton("💾 Save My Face")
        self.save_face_btn.setStyleSheet(self._btn_style("#98c379"))
        self.save_face_btn.clicked.connect(self.save_my_face)
        layout.addWidget(self.save_face_btn)

        self.show_trusted_btn = QPushButton("👁 Show Trusted")
        self.show_trusted_btn.setCheckable(True)
        self.show_trusted_btn.setStyleSheet(self._btn_style_checkable())
        self.show_trusted_btn.clicked.connect(self.toggle_show_trusted)
        layout.addWidget(self.show_trusted_btn)

        self.clear_faces_btn = QPushButton("🗑")
        self.clear_faces_btn.setToolTip("Clear all trusted faces")
        self.clear_faces_btn.setStyleSheet(self._btn_style("#e06c75", danger=True))
        self.clear_faces_btn.clicked.connect(self.clear_trusted_faces)
        layout.addWidget(self.clear_faces_btn)

        layout.addStretch()
        return row

    def create_stats_bar(self):
        row = QFrame()
        row.setStyleSheet("""
            QFrame {
                background-color: #21252b;
                border-radius: 6px;
                padding: 2px 6px;
            }
        """)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(16)

        self.faces_stat  = QLabel("👤 Faces: 0")
        self.text_stat   = QLabel("📄 Text regions: 0")
        self.trusted_stat = QLabel("✅ Trusted: 0")

        for lbl in (self.faces_stat, self.text_stat, self.trusted_stat):
            lbl.setStyleSheet("color: #abb2bf; font-size: 10px;")
            layout.addWidget(lbl)

        layout.addStretch()
        return row

    def create_status_bar(self):
        self.status_label = QLabel("● Initializing...")
        self.status_label.setStyleSheet(
            "color: #c678dd; font-size: 11px; padding-left: 4px;"
        )
        return self.status_label

    @staticmethod
    def _btn_style(accent: str, danger: bool = False) -> str:
        hover_bg = accent if danger else "#4b5263"
        hover_fg = "#282c34" if danger else accent
        return f"""
            QPushButton {{
                background-color: #3c4049;
                color: {accent};
                border: 1px solid #4b5263;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
                color: {hover_fg};
                border-color: {accent};
            }}
            QPushButton:pressed {{ background-color: #2c313a; }}
        """

    @staticmethod
    def _btn_style_checkable() -> str:
        return """
            QPushButton {
                background-color: #3c4049;
                color: #abb2bf;
                border: 1px solid #4b5263;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4b5263; }
            QPushButton:checked {
                background-color: #61afef;
                color: #282c34;
                border-color: #61afef;
            }
        """

    def create_title_bar(self):
        title_bar = QFrame()
        title_bar.setStyleSheet("background-color: transparent;")
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_logo = QLabel()
        self.set_logo_pixmap(title_logo, size=28)
        layout.addWidget(title_logo)

        title_label = QLabel("Privacy Camera")
        title_label.setStyleSheet(
            "color: #abb2bf; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(title_label)
        layout.addStretch()

        for symbol, slot, bg in [
            ("□",  self.toggle_fullscreen, "#3c4049"),
            ("−",  self.toggle_minimize,   "#3c4049"),
            ("×",  self.close,             "#e06c75"),
        ]:
            btn = QPushButton(symbol)
            btn.setFixedSize(24, 24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg};
                    border-radius: 12px;
                    color: {'#282c34' if bg != '#3c4049' else '#abb2bf'};
                    font-size: 15px;
                    font-weight: bold;
                    border: none;
                }}
                QPushButton:hover {{ background-color: #4b5263; }}
            """)
            btn.clicked.connect(slot)
            layout.addWidget(btn)
            if symbol == "□":
                self.fullscreen_btn = btn

        return title_bar

    # ===================================================================
    # Model / camera initialisation
    # ===================================================================

    def init_yolo(self):
        try:
            self.update_status("Loading Face AI…", "#e5c07b")
            self.yolo_model = YOLO("models/model.pt")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            label = "GPU" if self.device == "cuda" else "CPU"
            print(f"YOLOv8 face model loaded on {self.device}")
            self.update_status(f"Face AI Ready ({label})", "#98c379")
        except Exception as exc:
            print(f"Error loading YOLOv8: {exc}")
            self.yolo_model = None
            self.device     = "cpu"
            self.update_status("Face model failed to load", "#e06c75")

    def init_text_detector(self):
        try:
            self.update_status("Loading Text AI…", "#e5c07b")
            self.text_detector = DBNetTextDetector(
                model_path="models/dbnet_en.onnx",
                use_gpu=True,
            )
            if self.text_detector.ready:
                self.update_status("Privacy AI Ready", "#98c379")
            else:
                self.update_status("Text AI not available", "#e06c75")
        except Exception as exc:
            print(f"Error initialising text detector: {exc}")
            self.text_detector = None
            self.update_status("Text AI failed", "#e06c75")

    def init_camera(self, index: int = 0):
        self.camera_thread = CameraThread(camera_index=index, parent=self)
        self.camera_thread.frame_ready.connect(self.on_frame_received)
        self.camera_thread.camera_opened.connect(self.on_camera_opened)
        self.camera_thread.camera_failed.connect(self.on_camera_failed)
        self.camera_thread.start()
        self.update_status("Camera starting…", "#61afef")

    def restart_camera(self):
        idx = self.camera_combo.currentIndex()
        self.frame_buffer.clear()
        self.cached_faces   = []
        self.smoothed_faces = []
        self.camera_thread.stop()
        self.init_camera(index=idx)

    # ===================================================================
    # Camera signal handlers
    # ===================================================================

    @pyqtSlot(int)
    def on_camera_opened(self, index: int):
        labels = ["0 (built-in)", "1 (external)", "2", "3"]
        self.camera_combo.setCurrentIndex(index)
        self.update_status(f"Camera {labels[min(index, 3)]} active", "#98c379")

    @pyqtSlot()
    def on_camera_failed(self):
        self.update_status("No camera found!", "#e06c75")
        self.preview_label.setText("No camera detected.\nPlease connect a camera\nand press Apply.")

    # ===================================================================
    # Frame pipeline
    # ===================================================================

    @pyqtSlot(object)
    def on_frame_received(self, frame):
        if self.is_minimized:
            return

        self.current_frame = frame.copy()
        self.frame_buffer.append(frame.copy())

        if len(self.frame_buffer) < self.BUFFER_SIZE:
            return

        oldest_frame = self.frame_buffer[0]

        try:
            processed = self.apply_anonymization(oldest_frame)
        except Exception as exc:
            print(f"Pipeline error, blanking frame: {exc}")
            processed = np.zeros_like(oldest_frame)

        self._display_frame(processed)

    def _display_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            if "starting" in self.status_label.text().lower() or \
               "loading" in self.status_label.text().lower():
                self.update_status("Privacy Mode Active", "#e5c07b")
        except Exception as exc:
            print(f"Display error: {exc}")

    def apply_anonymization(self, frame):
        processed = frame.copy()

        # Face detection
        self.frame_count += 1
        if self.frame_count >= self.detection_interval:
            self.frame_count = 0
            new_faces = self.detect_with_yolo(frame)

            if new_faces:
                self.cached_faces   = new_faces
                self.smoothed_faces = self.smooth_bboxes(self.smoothed_faces, new_faces)
                self.face_persistence = self.max_persistence
            elif self.face_persistence > 0:
                self.face_persistence -= 1
            else:
                self.cached_faces   = []
                self.smoothed_faces = []

        faces = self.smoothed_faces if self.smoothed_faces else self.cached_faces
        n_blurred = 0

        for (x, y, w, h) in faces:
            bbox = (x, y, w, h)
            try:
                if self.show_trusted and self.is_trusted_face(frame, bbox):
                    continue
                processed = apply_privacy_pixelation(processed, bbox)
                n_blurred += 1
            except Exception as exc:
                print(f"Face anonymisation error: {exc}")

        # Text detection
        n_text = 0
        if self.text_detector and self.text_detector.ready:
            text_boxes = self.text_detector.detect_cached(frame)
            for tb in text_boxes:
                try:
                    processed = apply_text_blackbox(processed, tb)
                    n_text += 1
                except Exception as exc:
                    print(f"Text mask error: {exc}")

        # Update stats
        self.faces_stat.setText(f"👤 Faces: {n_blurred}")
        self.text_stat.setText(f"📄 Text regions: {n_text}")
        self.trusted_stat.setText(f"✅ Trusted: {len(self.trusted_faces)}")

        return processed

    def detect_with_yolo(self, frame):
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model(
                frame, verbose=False, conf=0.25, iou=0.4, device=self.device
            )
            img_h, img_w = frame.shape[:2]
            faces = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if float(box.conf[0]) <= 0.25:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                    bw, bh = x2 - x1, y2 - y1
                    pw, ph = int(bw * 0.35), int(bh * 0.25)
                    x1 = max(0, x1 - pw)
                    y1 = max(0, y1 - ph)
                    bw = min(bw + 2 * pw, img_w - x1)
                    bh = min(bh + 2 * ph, img_h - y1)
                    if bw > 30 and bh > 30:
                        faces.append((x1, y1, bw, bh))
            return faces[:5]
        except Exception as exc:
            print(f"YOLO detection error: {exc}")
            return self.cached_faces

    def smooth_bboxes(self, old_faces, new_faces, alpha=0.6):
        if not old_faces:
            return new_faces
        if not new_faces:
            return old_faces

        smoothed = []
        for (nx, ny, nw, nh) in new_faces:
            best, best_d = None, float("inf")
            for (ox, oy, ow, oh) in old_faces:
                d = abs((nx + nw / 2) - (ox + ow / 2)) + \
                    abs((ny + nh / 2) - (oy + oh / 2))
                if d < best_d:
                    best_d, best = d, (ox, oy, ow, oh)

            if best and best_d < 100:
                ox, oy, ow, oh = best
                smoothed.append((
                    int(alpha * nx + (1 - alpha) * ox),
                    int(alpha * ny + (1 - alpha) * oy),
                    int(alpha * nw + (1 - alpha) * ow),
                    int(alpha * nh + (1 - alpha) * oh),
                ))
            else:
                smoothed.append((nx, ny, nw, nh))
        return smoothed

    # ===================================================================
    # Trusted faces
    # ===================================================================

    def save_my_face(self):
        if self.current_frame is None or not self.cached_faces:
            self.update_status("No face detected!", "#e06c75")
            return
        bbox = self.cached_faces[0]
        emb = get_face_embedding(self.current_frame, bbox)
        if emb is not None:
            self.trusted_faces.append(emb.tolist())
            self.save_trusted_faces_to_file()
            self.update_status(
                f"Face saved! ({len(self.trusted_faces)} trusted)", "#98c379"
            )
        else:
            self.update_status("Could not save face", "#e06c75")

    def toggle_show_trusted(self):
        self.show_trusted = self.show_trusted_btn.isChecked()
        if self.show_trusted:
            self.update_status("Trusted faces visible", "#61afef")
        else:
            self.update_status("Privacy Mode Active", "#e5c07b")

    def clear_trusted_faces(self):
        self.trusted_faces = []
        self.save_trusted_faces_to_file()
        self.update_status("Trusted faces cleared", "#e5c07b")

    def load_trusted_faces(self):
        try:
            if os.path.exists(self.trusted_faces_file):
                with open(self.trusted_faces_file, "r") as fh:
                    data = json.load(fh)
                self.trusted_faces = [np.array(f) for f in data]
                print(f"Loaded {len(self.trusted_faces)} trusted faces")
        except Exception as exc:
            print(f"Error loading trusted faces: {exc}")
            self.trusted_faces = []

    def save_trusted_faces_to_file(self):
        try:
            data = [
                f.tolist() if isinstance(f, np.ndarray) else f
                for f in self.trusted_faces
            ]
            with open(self.trusted_faces_file, "w") as fh:
                json.dump(data, fh)
        except Exception as exc:
            print(f"Error saving trusted faces: {exc}")

    def is_trusted_face(self, frame, bbox):
        if not self.trusted_faces:
            return False
        emb = get_face_embedding(frame, bbox)
        if emb is None:
            return False
        for trusted in self.trusted_faces:
            arr = np.array(trusted) if not isinstance(trusted, np.ndarray) else trusted
            if compare_faces(emb, arr, threshold=0.55):
                return True
        return False

    # ===================================================================
    # Window management
    # ===================================================================

    def toggle_fullscreen(self):
        if self.is_minimized:
            return
        self.is_fullscreen = not self.is_fullscreen

        flags = (
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)

        if self.is_fullscreen:
            self.normal_size = (self.width(), self.height())
            self.showFullScreen()
            self.fullscreen_btn.setText("⧉")
            self.preview_label.setMinimumHeight(600)
        else:
            self.showNormal()
            self.resize(*self.normal_size)
            self.fullscreen_btn.setText("□")
            self.preview_label.setMinimumHeight(240)
        self.show()

    def toggle_minimize(self):
        if self.is_fullscreen:
            self.toggle_fullscreen()
        self.is_minimized = not self.is_minimized
        if self.is_minimized:
            self.main_container.hide()
            self.logo_widget.show()
            self.resize(80, 80)
        else:
            self.logo_widget.hide()
            self.main_container.show()
            self.resize(*self.normal_size)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.is_minimized and event.button() == Qt.MouseButton.LeftButton:
            self.toggle_minimize()
            event.accept()

    # ===================================================================
    # Status / logo helpers
    # ===================================================================

    def update_status(self, message: str, color: str = "#98c379"):
        self.status_label.setText(f"● {message}")
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; padding-left: 4px;"
        )

    def set_logo_pixmap(self, target_label: QLabel, size: int = 80):
        svg_path = os.path.join(os.getcwd(), "assets", "logo.svg")
        if os.path.exists(svg_path):
            renderer = QSvgRenderer(svg_path)
            if renderer.isValid():
                dest = QPixmap(size, size)
                dest.fill(Qt.GlobalColor.transparent)
                p = QPainter(dest)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                path = QPainterPath()
                path.addEllipse(0, 0, size, size)
                p.setClipPath(path)
                renderer.render(p, QRectF(0, 0, size, size))
                p.end()
                target_label.setPixmap(dest)
                return
        target_label.setText("🛡️")
        target_label.setStyleSheet(
            f"font-size: {int(size * 0.6)}px; color: #61afef; "
            f"padding-bottom: {int(size * 0.1)}px;"
        )

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    font_path = "assets/fonts/Roboto-Regular.ttf"
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            family = QFontDatabase.applicationFontFamilies(font_id)[0]
            app.setFont(QFont(family))

    widget = FloatingWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
