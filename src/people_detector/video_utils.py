"""
Утилиты для чтения/записи видео и отрисовки детекций.
"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Отрисовать детекции на кадре.

    Args:
        frame: BGR image (numpy.ndarray)
        detections: список словарей с ключами 'bbox' и 'score'

    Возврат:
        Изменённый кадр (копия).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        # Ограничим координаты внутри кадра
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Прямоугольник (граница)
        color = (14, 204, 59)  # BGR, яркий зелёный
        thickness = max(1, int(round(min(w, h) / 400)))  # масштабируем толщину
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Текст с уверенностью
        label = f"person {score:.2f}"
        # calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        ((text_w, text_h), _) = cv2.getTextSize(label, font, font_scale, 1)

        # Фон для текста (полупрозрачный)
        text_bg_tl = (x1, max(0, y1 - text_h - 6))
        text_bg_br = (x1 + text_w + 6, y1)
        overlay = out.copy()
        cv2.rectangle(overlay, text_bg_tl, text_bg_br, color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Рисуем текст поверх
        text_org = (x1 + 3, y1 - 4)
        cv2.putText(out, label, text_org, font, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return out


def process_video(input_path: str, output_path: str, detector, skip_frames: int = 1) -> None:
    """
    Прочитать видео, прогнать детектор по каждому кадру и записать результат.

    Args:
        input_path: путь к входному видео
        output_path: путь к выходному видео (mp4)
        detector: экземпляр класса с методом predict(frame)->List[Dict]
        skip_frames: обрабатывать каждый N-ый кадр (по умолчанию 1)
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Видео пустое или не удалось прочитать первый кадр.")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Прогресс-бар
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    frame_idx = 0
    # Если мы уже прочитали первый кадр, вернём его в цикл
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            detections = detector.predict(frame)
            vis = draw_detections(frame, detections)
        else:
            vis = frame

        out.write(vis)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
