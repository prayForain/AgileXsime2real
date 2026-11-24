#!/usr/bin/env python3
"""Quick test script to verify the Orbbec Dabai DC1 wrist RGB camera via OpenCV."""

import argparse
import os
import time
from pathlib import Path
import cv2

DEFAULT_DEV = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1Z8620073-video-index0"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "debug" / "wrist_cam_snapshot.jpg")


def main():
    parser = argparse.ArgumentParser(description="Capture a test frame from the PiperX wrist camera")
    parser.add_argument("--device", default=DEFAULT_DEV, help="V4L2 device path or index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output image path",
    )
    args = parser.parse_args()

    dev = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: Could not open camera device: {args.device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Warm up auto-exposure
    frame = None
    t0 = time.monotonic()
    frames_ok = 0
    for _ in range(15):
        ret, img = cap.read()
        if ret and img is not None:
            frame = img
            frames_ok += 1
    elapsed = time.monotonic() - t0

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frame is None:
        raise SystemExit("ERROR: Opened camera but failed to capture any frame.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"Camera OK! Device: {args.device}")
    print(f"Resolution: {actual_w}x{actual_h} | Reported FPS: {actual_fps:.1f} | Measured: {frames_ok/elapsed:.1f} Hz")
    print(f"Saved snapshot to: {args.out}")


if __name__ == "__main__":
    main()
