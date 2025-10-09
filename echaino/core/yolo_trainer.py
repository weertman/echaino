# FILE: core\yolo_trainer.py
# PATH: D:\urchinScanner\core\yolo_trainer.py

from pathlib import Path
from typing import List, Dict, Tuple
from data.models import Segmentation
from data.archive_manager import ArchiveManager
from core.tile_manager import AdaptiveTileManager  # Added import
from utils.geometry_utils import polygon_to_yolo_format, clip_polygon_to_bbox
from utils.image_utils import MemoryEfficientImageLoader
import yaml
import random
from datetime import datetime  # Added for unique timestamps
from shapely.geometry import Polygon  # Added import for Polygon
import cv2

import logging

logger = logging.getLogger(__name__)


class YoloTrainer:
    """Handles YOLO dataset preparation and training from archives"""

    def __init__(self, config: Dict, archive_manager: ArchiveManager):  # Updated: Accept archive_manager
        self.config = config
        self.archive_manager = archive_manager  # Added
        self.tile_manager = AdaptiveTileManager(config)  # Added: Initialize tile_manager
        self.min_clip_area = config['yolo'].get('min_clip_area', 10)

    def prepare_dataset(self, selected_archives: List[str], params: Dict) -> Path:
        """Prepare YOLO dataset from selected archives"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp

        dataset_root = Path(self.config['yolo']['dataset_output_dir'])
        dataset_root.mkdir(parents=True, exist_ok=True)

        num_archives = len(selected_archives)
        dataset_name = f"combined_{num_archives}_archives_{timestamp}"  # Unique name
        dataset_path = dataset_root / dataset_name
        dataset_path.mkdir()

        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        for d in [images_dir, labels_dir]:
            d.mkdir()

        train_images = images_dir / 'train'
        val_images = images_dir / 'val'
        train_labels = labels_dir / 'train'
        val_labels = labels_dir / 'val'
        for d in [train_images, val_images, train_labels, val_labels]:
            d.mkdir(parents=True)

        all_tiles: List[Tuple[Path, Path]] = []  # (image_path, label_path)

        for archive in selected_archives:
            json_files = self.archive_manager.get_sorted_json_files(archive)  # Updated: Use instance
            if not json_files:
                continue

            latest_json = json_files[-1]
            project = self.archive_manager.load_project(latest_json)  # Updated: Use instance
            if not project or not project.segmentations:
                continue

            # Get absolute image path
            absolute_image_path = self.archive_manager.get_absolute_image_path(archive, project.image_path)  # Added: Ensure absolute

            image_loader = MemoryEfficientImageLoader(absolute_image_path)  # Updated
            full_image = image_loader.load_full_image()
            img_h, img_w = full_image.shape[:2]

            tiles = self.tile_manager.generate_fixed_tiles((img_h, img_w), project.segmentations, params['img_size'])

            for tile in tiles:
                tile_image = full_image[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
                tile_filename = f"{archive}_{tile.id}.jpg"
                tile_image_path = images_dir / tile_filename
                cv2.imwrite(str(tile_image_path), cv2.cvtColor(tile_image, cv2.COLOR_RGB2BGR))

                label_path = labels_dir / f"{archive}_{tile.id}.txt"
                with open(label_path, 'w') as f:
                    for seg in tile.segmentations:
                        clipped_poly = clip_polygon_to_bbox(
                            seg.polygon,
                            (tile.x, tile.y, tile.width, tile.height)
                        )
                        if len(clipped_poly) < 3:
                            continue

                        translated_poly = [
                            [px - tile.x, py - tile.y] for px, py in clipped_poly
                        ]

                        area = Polygon(translated_poly).area
                        if area < params.get('min_clip_area', self.min_clip_area):
                            continue

                        yolo_line = polygon_to_yolo_format(translated_poly, tile.width, tile.height, class_id=seg.class_id)  # NEW: Pass class_id
                        f.write(yolo_line + '\n')

                all_tiles.append((tile_image_path, label_path))

        random.shuffle(all_tiles)
        split_idx = int(len(all_tiles) * params['train_val_split'])
        train_tiles = all_tiles[:split_idx]
        val_tiles = all_tiles[split_idx:]

        def move_files(tiles: List[Tuple[Path, Path]], img_dest: Path, label_dest: Path):
            for img_src, label_src in tiles:
                img_dest_file = img_dest / img_src.name
                label_dest_file = label_dest / label_src.name
                img_src.rename(img_dest_file)
                label_src.rename(label_dest_file)

        move_files(train_tiles, train_images, train_labels)
        move_files(val_tiles, val_images, val_labels)

        # NEW: Multi-class YAML
        classes = self.config.get('classes', ['urchin'])  # Changed fallback from ['object'] to ['urchin'] for compat
        nc = len(classes)
        names = classes
        data_yaml = {
            'train': str(train_images.relative_to(dataset_path)),
            'val': str(val_images.relative_to(dataset_path)),
            'nc': nc,
            'names': names
        }
        with open(dataset_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

        logger.info(f"Dataset prepared at {dataset_path}")
        logger.info(f"Train: {len(train_tiles)} images, Val: {len(val_tiles)} images")

        return dataset_path

    def train_model(self, dataset_path: Path, params: Dict, timestamp: str) -> Path:  # Accept timestamp
        """Train YOLO model on prepared dataset"""
        from ultralytics import YOLO

        model = YOLO(self.config['yolo']['pretrained_model'])

        project_dir = Path(self.config['yolo']['models_dir']) / f"echaino_model_{timestamp}"  # Unique name

        results = model.train(
            data=str(dataset_path / 'data.yaml'),
            epochs=params['epochs'],
            batch=params['batch_size'],
            imgsz=params['img_size'],
            device=self.config['yolo']['device'],
            project=str(project_dir),
            name="train",
            exist_ok=True
        )

        best_model_path = results.save_dir / 'weights' / 'best.pt'
        return best_model_path