#!/usr/bin/env python3
"""
Точка входа для запуска детекции людей в видео.

Пример:
    python main.py --input crowd.mp4 --output outputs/result.mp4 --confidence 0.6 --device cpu
"""

import argparse
from pathlib import Path

from src.people_detector.detector import PeopleDetector
from src.people_detector.video_utils import process_video


def parse_args():
    """Парсит аргументы командной строки и возвращает Namespace."""
    parser = argparse.ArgumentParser(description="People detection in video")
    # parser.add_argument(
    #     "--input", "-i", required=True, help="Path to input video (e.g. crowd.mp4)"
    # )
    parser.add_argument(
        "--input", "-i", default="crowd.mp4", help="Path to input video (default: crowd.mp4)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/result.mp4",
        help="Path to save output video (default: outputs/result.mp4)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="Device to run model on: 'cpu' or 'cuda' (if available)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every N-th frame (useful to speed up processing), default=1",
    )
    return parser.parse_args()


def main():
    """Точка входа: создаёт детектор и запускает обработку видео."""
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = PeopleDetector(device=args.device, confidence=args.confidence)
    process_video(
        input_path=str(input_path),
        output_path=str(output_path),
        detector=detector,
        skip_frames=max(1, args.skip),
    )


if __name__ == "__main__":
    main()
