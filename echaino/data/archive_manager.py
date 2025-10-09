# FILE: data\archive_manager.py
# PATH: D:\urchinScanner\data\archive_manager.py

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import os
import logging
import json
import shutil  # Added for file copy

from data.project_manager import ProjectManager
from data.models import ProjectData

logger = logging.getLogger(__name__)


class ArchiveManager:
    """Manages project archives with timestamped subfolders"""

    def __init__(self, config: dict, project_manager: ProjectManager):
        self.config = config
        self.root_dir = Path(config['archive']['root_dir'])
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.subfolder_format = config['archive']['subfolder_format']
        self.project_manager = project_manager

    def create_archive_for_image(self, image_path: str) -> str:
        """Create new archive subfolder and copy image"""
        image_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder = self.subfolder_format.format(timestamp=timestamp, image_name=image_name)
        archive_dir = self.root_dir / subfolder
        archive_dir.mkdir(parents=True)

        # Copy image
        dest_path = archive_dir / Path(image_path).name
        shutil.copy2(image_path, dest_path)  # Use shutil for cross-platform copy

        logger.info(f"Created archive: {subfolder}")
        return subfolder

    def find_existing_for_image(self, image_name: str) -> Optional[str]:
        """Find existing archive containing image_name"""
        for subdir in self.root_dir.iterdir():
            if subdir.is_dir() and any(f.stem == image_name for f in subdir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.tif']):
                return subdir.name
        return None

    def save_data_products(self, project: ProjectData, subfolder: str, is_auto: bool = False):
        """Save project JSON to archive"""
        archive_dir = self.root_dir / subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = '_auto' if is_auto else ''
        json_path = archive_dir / f"project_{timestamp}{suffix}.json"
        self.project_manager.save_project(project, str(json_path))
        logger.info(f"Saved project to {json_path}")

    def list_archives(self) -> List[str]:
        """List all archive subfolders"""
        return [d.name for d in self.root_dir.iterdir() if d.is_dir()]

    def get_latest_json(self, subfolder: str) -> Optional[Path]:
        """Get most recent JSON in archive"""
        json_files = self.get_sorted_json_files(subfolder)
        return json_files[-1] if json_files else None

    def get_sorted_json_files(self, subfolder: str) -> List[Path]:
        """Get sorted list of JSON files in archive by modification time"""
        archive_dir = self.root_dir / subfolder
        json_files = list(archive_dir.glob("project_*.json"))
        json_files.sort(key=lambda p: os.path.getmtime(p))
        return json_files

    def load_project(self, json_path: Path) -> Optional[ProjectData]:
        """Load project from JSON using ProjectManager"""
        return self.project_manager.load_project(str(json_path))

    def get_absolute_image_path(self, subfolder: str, relative_path: str) -> str:
        """Get absolute path to image in archive"""
        archive_dir = self.root_dir / subfolder
        image_name = Path(relative_path).name
        image_path = archive_dir / image_name
        if not image_path.exists():
            logger.warning(f"Image not found in archive: {image_path}")
            return ""
        return str(image_path.absolute())