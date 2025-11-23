import sys
import os
from pathlib import Path

# Add the backend to path
backend_dir = Path("task-3/backend")
sys.path.insert(0, str(backend_dir))

try:
    from app.services.model import RatingModelService
    from app.config import get_settings

    settings = get_settings()
    print(f"Task2 dir: {settings.task2_dir}")
    print(f"Model file: {settings.task2_dir / settings.rating_model_filename}")
    print(f"Model exists: {(settings.task2_dir / settings.rating_model_filename).exists()}")

    # Try to create the service (this will trigger the model loading)
    service = RatingModelService(settings, None)
    print("✅ Model loaded successfully!")
    print(f"Model version: {service._model_version}")
    print(f"Embedding cols: {len(service._embedding_cols)} columns")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
