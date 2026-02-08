"""
Food recognition model loader and inference
Uses BinhQuocNguyen/food-recognition-model (101 food categories)
Using transformers pipeline for simpler loading
"""

from transformers import pipeline
from PIL import Image
import io
import base64
import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

class FoodRecognitionModel:
    """
    Food recognition model wrapper
    Loads and runs inference using transformers pipeline
    """

    def __init__(self, model_name: str = None):
        """
        Initialize model

        Args:
            model_name: Hugging Face model name. Defaults to BinhQuocNguyen/food-recognition-model
        """
        self.model_name = model_name or os.getenv(
            "MODEL_NAME",
            "BinhQuocNguyen/food-recognition-model"
        )

        logger.info(f"Loading model: {self.model_name}")

        try:
            # Load model using transformers pipeline
            self.classifier = pipeline(
                "image-classification",
                model=self.model_name
            )

            logger.info(f"Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _decode_base64_image(self, base64_data: str) -> Image.Image:
        """
        Decode base64 data URL to PIL Image

        Args:
            base64_data: Base64 data URL (data:image/jpeg;base64,...)

        Returns:
            PIL Image object
        """
        try:
            # Remove data URL prefix if present
            if base64_data.startswith("data:image"):
                # Split on comma to get actual base64 data
                base64_data = base64_data.split(",", 1)[1]

            # Decode base64
            image_bytes = base64.b64decode(base64_data)

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (handle RGBA, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")

    def predict(self, base64_data: str, top_k: int = 20) -> List[Dict[str, any]]:
        """
        Run inference on base64 encoded image

        Args:
            base64_data: Base64 data URL of the image
            top_k: Number of top predictions to return (default: 20)

        Returns:
            List of classifications [{"label": str, "score": float}, ...]
        """
        try:
            # Decode image
            image = self._decode_base64_image(base64_data)

            # Run inference using pipeline
            results = self.classifier(image, top_k=top_k)

            # Format results to match expected output
            classifications = [
                {
                    "label": result["label"],
                    "score": result["score"]
                }
                for result in results
            ]

            logger.info(f"Top prediction: {classifications[0]['label']} ({classifications[0]['score']:.2%})")

            return classifications

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "status": "ready"
        }


# Standalone testing
if __name__ == "__main__":
    # Test with a sample image
    print("Testing food recognition model...")

    model = FoodRecognitionModel()
    print(f"Model loaded: {model.model_name}")

    # Example: You can test with a base64 encoded image
    # classifications = model.predict("data:image/jpeg;base64,...")
    # print(classifications)
