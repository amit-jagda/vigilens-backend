import io
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from insightface.app import FaceAnalysis

class FaceRecognitionService:
    def __init__(self):
        self.apps = {}

    def _lazy_init(self, model_name: str = "buffalo_l"):
        if model_name not in self.apps:
            register_heif_opener()
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")

                app = FaceAnalysis(
                    name=model_name,
                    providers=providers
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                self.apps[model_name] = app
            except Exception as e:
                import sys
                print(
                    f"\n[FaceRecognitionService ERROR] Failed to initialize InsightFace FaceAnalysis for model '{model_name}': {e}",
                    file=sys.stderr
                )
                raise e

    def extract_faces(self, image_bytes: bytes, model_name: str = "buffalo_l") -> list[dict]:
        """
        Decodes image bytes and extracts bounding boxes and 512D embeddings.
        """
        self._lazy_init(model_name)
        app = self.apps[model_name]
        
        img = None
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_rgb = np.array(image)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image file format or corrupted image bytes.")

        faces = app.get(img)
        
        results = []
        for i, face in enumerate(faces):
            results.append({
                "face_idx": i,
                "bbox": [int(x) for x in face.bbox],
                "embedding": face.embedding.tolist(),
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0,
                "kps": face.kps.tolist() if getattr(face, "kps", None) is not None else None,
                "gender": int(face.gender) if getattr(face, "gender", None) is not None else None,
                "age": int(face.age) if getattr(face, "age", None) is not None else None
            })
            
        return results

    def extract_faces_from_image(self, img_bgr: np.ndarray, model_name: str = "buffalo_l") -> list[dict]:
        """
        Extracts bounding boxes and embeddings for all detected faces directly from a BGR numpy array.
        Zero-copy, avoids in-memory byte re-encoding.
        """
        if img_bgr is None or img_bgr.size == 0 or img_bgr.shape[0] < 10 or img_bgr.shape[1] < 10:
            return []

        self._lazy_init(model_name)
        app = self.apps[model_name]

        # Detect faces and extract embeddings
        faces = app.get(img_bgr)

        results = []
        for i, face in enumerate(faces):
            results.append({
                "face_idx": i,
                "bbox": [int(x) for x in face.bbox],
                "embedding": face.embedding.tolist(),
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0,
                "kps": face.kps.tolist() if getattr(face, "kps", None) is not None else None,
                "gender": int(face.gender) if getattr(face, "gender", None) is not None else None,
                "age": int(face.age) if getattr(face, "age", None) is not None else None
            })

        return results

# Self-contained singleton instance
face_rec_service = FaceRecognitionService()
