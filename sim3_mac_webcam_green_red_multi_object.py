import cv2
import numpy as np
import math
import time
from collections import deque

# HSV ranges
LOWER_GREEN = np.array([35, 80, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Red wraps around both ends of OpenCV HSV hue; thresholds are relaxed to recover red detections.
LOWER_RED_1 = np.array([0, 100, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 100, 50])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 500

# Tracking settings (no centre-history trail is drawn).
TRAIL_LENGTH = 150
MOVEMENT_THRESHOLD = 5
STATIONARY_TIME = 0.7

# macOS webcam capture using OpenCV AVFoundation.
# This version tests vision logic on a Mac; it does not use the Pi AI Camera.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("ERROR: Could not open Mac webcam.")
    print("Check macOS camera permissions for Terminal or VS Code.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Mac webcam opened. Waiting for frames...")
frame = None
for i in range(10):
    ret, captured = cap.read()
    if ret and captured is not None and captured.size > 0:
        frame = captured
        print("First valid webcam frame received!")
        break
    time.sleep(0.3)

if frame is None:
    print("ERROR: No frames received after 10 attempts.")
    cap.release()
    raise SystemExit(1)

# Use the actual captured frame size; don't reconfigure the active stream.
frame_height, frame_width = frame.shape[:2]
print("--------------------------------------")
print("Camera started")
print(f"Resolution: {frame_width} x {frame_height}")
print("--------------------------------------")

kernel = np.ones((5, 5), np.uint8)

tracking = {
    "GREEN": {
        "history": deque(maxlen=TRAIL_LENGTH),
        "previous_cx": None,
        "previous_cy": None,
        "last_movement_time": time.time(),
        "stationary": False,
    },
    "RED": {
        "history": deque(maxlen=TRAIL_LENGTH),
        "previous_cx": None,
        "previous_cy": None,
        "last_movement_time": time.time(),
        "stationary": False,
    },
}

# OpenCV drawing colours are BGR.
OBJECT_COLORS = {"GREEN": (0, 255, 0), "RED": (0, 0, 255)}

try:
    consecutive_read_failures = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            consecutive_read_failures += 1
            print(f"WARNING: Could not read frame ({consecutive_read_failures}/10).")
            if consecutive_read_failures >= 10:
                print("ERROR: Webcam stream failed repeatedly; stopping.")
                break
            time.sleep(0.1)
            continue

        consecutive_read_failures = 0
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green segmentation
        green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # Red segmentation (red occupies both ends of the HSV hue scale)
        red_mask_1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        red_mask_2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Reduce small noise in both masks.
        green_mask = cv2.erode(green_mask, kernel, iterations=1)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        # Colour-coded mask window.
        combined_mask = np.zeros_like(frame)
        combined_mask[green_mask > 0] = OBJECT_COLORS["GREEN"]
        combined_mask[red_mask > 0] = OBJECT_COLORS["RED"]

        detections = {"GREEN": green_mask, "RED": red_mask}
        hud_info = {}

        # Process EVERY contour above the minimum area, not just the largest.
        for color_name, mask in detections.items():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            draw_color = OBJECT_COLORS[color_name]
            valid_contours = sorted(
                (c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA),
                key=cv2.contourArea,
                reverse=True,
            )

            objects = []
            for object_index, contour in enumerate(valid_contours, start=1):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                offset_x = cx - frame_center_x
                offset_y = cy - frame_center_y

                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), draw_color, 2)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), draw_color, 2)
                cv2.putText(
                    frame, f"{color_name} #{object_index}",
                    (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, draw_color, 2, cv2.LINE_AA
                )

                if color_name == "RED":
                    danger_padding = 18
                    dx1 = max(0, x - danger_padding)
                    dy1 = max(0, y - danger_padding)
                    dx2 = min(frame_width - 1, x + w + danger_padding)
                    dy2 = min(frame_height - 1, y + h + danger_padding)
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, "DANGER ZONE", (dx1, max(18, dy1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2,
                        cv2.LINE_AA
                    )

                objects.append({
                    "index": object_index,
                    "centre": (cx, cy),
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "area": int(area),
                })

            hud_info[color_name] = objects

        # HUD shows the number of detections and details for up to four objects per colour.
        hud_y = 25
        for color_name in ("GREEN", "RED"):
            draw_color = OBJECT_COLORS[color_name]
            objects = hud_info[color_name]
            lines = [f"{color_name}: {len(objects)} object(s)"]
            for info in objects[:4]:
                lines.append(
                    f"#{info['index']} C{info['centre']} "
                    f"XY({info['offset_x']:+d},{info['offset_y']:+d}) "
                    f"A{info['area']}px2"
                )
            if len(objects) > 4:
                lines.append(f"... +{len(objects) - 4} more")
            for line in lines:
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                cv2.rectangle(
                    frame, (8, hud_y - text_h - 3),
                    (16 + text_w, hud_y + baseline), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, line, (12, hud_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, draw_color, 1, cv2.LINE_AA
                )
                hud_y += 20
            hud_y += 5

        # Camera centre marker.
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)
        cv2.line(
            frame, (frame_center_x - 20, frame_center_y),
            (frame_center_x + 20, frame_center_y), (255, 0, 0), 1
        )
        cv2.line(
            frame, (frame_center_x, frame_center_y - 20),
            (frame_center_x, frame_center_y + 20), (255, 0, 0), 1
        )

        cv2.imshow("1 - HSV Color Identification", combined_mask)
        cv2.imshow("2 - Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Mac webcam stopped.")
