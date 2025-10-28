# train_yolo_script.py
import argparse
import os
import shutil
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument('--dataset_path', required=True, help="Path to dataset")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--imgsz', type=int, required=True, help="Image size")
    parser.add_argument('--device', required=True, help="Device (cpu/cuda)")
    parser.add_argument('--pretrained_model', required=True, help="Pretrained model path")
    parser.add_argument('--models_dir', required=True, help="Models directory")

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {vars(args)}")

    try:
        logger.info("Loading YOLO model")
        model = YOLO(args.pretrained_model)
        logger.info("Model loaded successfully")

        logger.info("Starting training")
        results = model.train(
            data=os.path.join(args.dataset_path, 'dataset.yaml'),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            name='small_object_model',
            workers=0
        )
        logger.info("Training completed")

        model_path = os.path.join(args.models_dir, 'best.pt')
        shutil.copy(results.save_dir / 'weights/best.pt', model_path)
        logger.info(f"Model saved to {model_path}")

        # Output model path as last line for parent process
        print(model_path)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    logger.info("Script started under __name__ == '__main__'")
    main()
    logger.info("Script completed")
