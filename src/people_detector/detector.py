"""
PeopleDetector: обёртка для модели детекции людей (Faster R-CNN из torchvision).

Класс загружает предобученную модель и предоставляет метод predict(frame),
возвращающий список детекций (bbox + score).
"""

from typing import List, Dict
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class PeopleDetector:
    """
    Класс-детектор людей.

    Args:
        device: 'cpu' или 'cuda'
        confidence: порог уверенности для отбора предсказаний
    """

    def __init__(self, device: str = "cpu", confidence: float = 0.5) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.confidence = float(confidence)
        # Загружаем предобученную модель Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frame: np.ndarray) -> List[Dict]:
        """
        Выполнить инференс на одном кадре.

        Args:
            frame: numpy.ndarray в формате BGR (как возвращает OpenCV)

        Returns:
            Список словарей: [{"bbox": (x1, y1, x2, y2), "score": float}, ...]
            только для класса "person" и с score >= self.confidence
        """
        # Преобразуем BGR->RGB и в PIL Image
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        # В тензор (0..1)
        tensor = F.to_tensor(pil_img).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])[0]

        boxes = outputs.get("boxes", [])
        labels = outputs.get("labels", [])
        scores = outputs.get("scores", [])

        results = []
        for box, label, score in zip(boxes, labels, scores):
            # В COCO label 1 == person
            if int(label.item()) == 1 and float(score.item()) >= self.confidence:
                x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
                results.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "score": float(score.item())})
        return results
