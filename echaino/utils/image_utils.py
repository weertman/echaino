import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class MemoryEfficientImageLoader:
    """Load only required image regions"""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self._shape = None
        self._thumbnail = None
        self._thumbnail_size = 1024

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape without loading full image"""
        if self._shape is None:
            with Image.open(self.image_path) as img:
                w, h = img.size
                channels = len(img.getbands())
                self._shape = (h, w, channels)
        return self._shape

    def load_full_image(self) -> np.ndarray:
        """Load complete image"""
        return cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)

    def load_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        try:
            with Image.open(self.image_path) as img:
                x2 = min(x + w, img.width)
                y2 = min(y + h, img.height)
                region = img.crop((x, y, x2, y2))
                return np.array(region)
        except MemoryError:  # IMPROVED: Handle large images
            logger.warning("Memory error; falling back to chunked load")
            # Chunked load example (simplified)
            full_img = cv2.imread(self.image_path, cv2.IMREAD_REDUCED_COLOR_2)  # Reduced for memory
            return cv2.resize(full_img[y:y + h, x:x + w], (w, h))  # Resize back
        except Exception as e:
            logger.error(f"Error loading image region: {e}")
            full_img = self.load_full_image()
            return full_img[y:y + h, x:x + w]

    def get_thumbnail(self, max_size: int = 1024) -> np.ndarray:
        """Get cached thumbnail for display"""
        if self._thumbnail is None or self._thumbnail_size != max_size:
            self._thumbnail_size = max_size
            img = cv2.imread(self.image_path)

            # Calculate scaling
            h, w = img.shape[:2]
            scale = min(max_size / w, max_size / h)

            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                self._thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                self._thumbnail = img

        return self._thumbnail


def create_overlay(image: np.ndarray, mask: np.ndarray,
                   color: Tuple[int, int, int], alpha: float = 0.3) -> np.ndarray:
    """Create colored overlay from mask"""
    overlay = image.copy()

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend with original image
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

    return result


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i * 360 / n
        # Convert HSV to RGB
        color = cv2.cvtColor(
            np.array([[[hue, 255, 255]]], dtype=np.uint8),
            cv2.COLOR_HSV2RGB
        )[0, 0]
        colors.append(tuple(int(c) for c in color))
    return colors

