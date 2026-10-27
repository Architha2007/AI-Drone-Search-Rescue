"""Real AI-based survivor detection module using Ultralytics YOLO and OpenCV.

Decoupled from core simulation so the application runs without YOLO or an internet connection.
"""
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BoundingBox:
    """Bounding box detection in image pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box (u, v)."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class YOLODetector:
    """
    YOLO-based computer vision detector for identifying human survivors in aerial imagery.

    Safety & Connectivity Policy:
    - Never automatically downloads weights without explicit user authorization.
    - Gracefully handles missing packages (ultralytics, opencv) and missing model weights.
    - Keeps core simulation fully functional even if vision dependencies are absent.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model: Any = None
        self.is_loaded: bool = False
        self.status_message: str = "Uninitialized"

        # Check dependency availability
        self.has_cv2: bool = False
        self.has_yolo: bool = False

        try:
            import cv2
            self.has_cv2 = True
        except ImportError:
            self.has_cv2 = False

        try:
            import ultralytics
            self.has_yolo = True
        except ImportError:
            self.has_yolo = False

        self._check_status()

    def _check_status(self) -> None:
        """Inspect environment and update status message."""
        if not self.has_cv2 and not self.has_yolo:
            self.status_message = "OpenCV and Ultralytics not installed. Vision mode disabled."
        elif not self.has_cv2:
            self.status_message = "OpenCV not installed. Install with 'pip install opencv-python'."
        elif not self.has_yolo:
            self.status_message = "Ultralytics YOLO not installed. Install with 'pip install ultralytics'."
        else:
            if os.path.exists(self.model_path):
                self.status_message = f"Weights found at '{self.model_path}'. Ready to load."
            else:
                self.status_message = (
                    f"Weights '{self.model_path}' not found locally. "
                    "Provide a local weights file or enable explicit download."
                )

    def load_model(self, allow_download: bool = False) -> bool:
        """
        Load the YOLO model.
        To avoid unexpected internet usage, downloads are blocked unless allow_download=True.
        """
        if not self.has_yolo:
            self.status_message = "Cannot load model: ultralytics package is missing."
            return False

        if not os.path.exists(self.model_path) and not allow_download:
            self.status_message = (
                f"Model weights '{self.model_path}' not found locally. "
                "Download blocked to preserve offline operation. Pass allow_download=True to download."
            )
            return False

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            self.status_message = f"Successfully loaded YOLO model: {self.model_path}"
            return True
        except Exception as e:
            self.is_loaded = False
            self.status_message = f"Failed to load model: {str(e)}"
            return False

    def detect_person(
        self,
        image_path: str,
    ) -> Tuple[Optional[Any], List[BoundingBox], str]:
        """
        Run inference on a local image and detect persons (COCO class 0).

        Returns:
            Tuple of:
            - annotated_image: OpenCV image (numpy array) with rendered bounding boxes, or None on error.
            - detections: List of BoundingBox objects for detected persons.
            - message: Status or error description.
        """
        if not self.has_cv2:
            return None, [], "OpenCV is not available."

        import cv2

        if not os.path.exists(image_path):
            return None, [], f"Image file not found: '{image_path}'"

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, [], f"Failed to decode image from '{image_path}'"

        if not self.is_loaded:
            return image, [], f"YOLO model not loaded. {self.status_message}"

        try:
            # Run inference targeting class 0 ('person' in COCO dataset)
            results = self.model.predict(
                source=image,
                classes=[0],  # Filter for 'person' class only
                conf=self.conf_threshold,
                verbose=False,
            )

            detections: List[BoundingBox] = []
            annotated_image = image.copy()

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names.get(cls_id, "person")

                    x1, y1, x2, y2 = map(int, xyxy)
                    detection = BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                    detections.append(detection)

                    # Draw bounding box and label
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            msg = f"Detection complete. Found {len(detections)} person(s)."
            return annotated_image, detections, msg

        except Exception as e:
            return image, [], f"Inference error: {str(e)}"

    @staticmethod
    def georeference_detection(
        box: BoundingBox,
        drone_grid_pos: Tuple[int, int],
        drone_altitude_m: float = 20.0,
        fov_horizontal_deg: float = 60.0,
        image_dims: Tuple[int, int] = (640, 480),
        grid_cell_size_m: float = 5.0,
    ) -> Tuple[int, int]:
        """
        Georeferencing / Simulated Position Association:
        Projects a 2D image bounding box center into 2D disaster grid coordinates.

        Mathematical Formulation:
        Let (u, v) be the pixel coordinate of the detection center in an image of size (W, H).
        The camera optical center is at (cx, cy) = (W/2, H/2).
        For a nadir (downward-facing) pinhole camera at drone position (X_drone, Y_drone) and altitude Z:
            Ground footprint width: G_w = 2 * Z * tan(FOV_h / 2)
            Meters per pixel: s = G_w / W
            Ground offset:
                delta_x = (u - cx) * s
                delta_y = (v - cy) * s
            World position (meters):
                X_world = X_drone_m + delta_x
                Y_world = Y_drone_m + delta_y
            Grid cell:
                grid_x = round(X_world / grid_cell_size_m)
                grid_y = round(Y_world / grid_cell_size_m)

        In a physical deployment, this requires IMU attitude angles (roll, pitch, yaw),
        GPS coordinates, camera intrinsic calibration matrix K, and digital elevation model (DEM).
        """
        import math

        img_w, img_h = image_dims
        u, v = box.center

        # Ground footprint width in meters
        fov_rad = math.radians(fov_horizontal_deg)
        ground_width_m = 2.0 * drone_altitude_m * math.tan(fov_rad / 2.0)
        meters_per_pixel = ground_width_m / float(img_w)

        # Offsets from image center
        delta_x_m = (u - (img_w / 2.0)) * meters_per_pixel
        delta_y_m = (v - (img_h / 2.0)) * meters_per_pixel

        # Convert drone grid position to world meters
        drone_world_x = drone_grid_pos[0] * grid_cell_size_m
        drone_world_y = drone_grid_pos[1] * grid_cell_size_m

        detected_world_x = drone_world_x + delta_x_m
        detected_world_y = drone_world_y + delta_y_m

        grid_x = int(round(detected_world_x / grid_cell_size_m))
        grid_y = int(round(detected_world_y / grid_cell_size_m))

        return (grid_x, grid_y)
