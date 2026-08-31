import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from typing import Optional

from modules.objectcount.osnet import osnet_x0_5

class ReIDService:
    """
    Singleton Re-ID feature extraction service wrapping OSNet x0.5.
    Generates normalized 512-dimensional visual embeddings from body crops.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load OSNet x0.5 pre-trained feature extractor
        self.feature_extractor = osnet_x0_5(pretrained=True)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Standard Re-ID crop transform (256x128)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a normalized 512-dimensional visual embedding from a person crop (BGR).
        Returns None if the crop is invalid or extraction fails.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            features = self.feature_extractor(tensor)
            
            # L2 normalize vector
            features = nn.functional.normalize(features, p=2, dim=1)
            features_np = features.squeeze().cpu().numpy().astype(np.float32)
            return features_np
        except Exception as e:
            print(f"[ReIDService] Embedding extraction error: {e}")
            return None

# Singleton instance
reid_service = ReIDService()
