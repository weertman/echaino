#!/usr/bin/env python
"""
Small Object Annotator - Main Entry Point
"""

import sys
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='EchAIno Annotator')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--image', type=str, help='Initial image to load')
    args = parser.parse_args()

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("EchAIno Annotator")
    app.setOrganizationName("EchAIno")

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = MainWindow()
    window.show()

    # Load initial image if provided
    if args.image and Path(args.image).exists():
        window.load_image(args.image)

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()