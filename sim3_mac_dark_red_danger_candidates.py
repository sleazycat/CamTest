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

# SAFETY: HSV red is only a candidate cue, never a confirmed restricted zone.
# With no fixed marker specification, this script intentionally does NOT
# draw/emit a DANGER ZONE from raw red segmentation alone.
RED_MIN_SATURATION = 125
RED_MIN_VALUE = 65
# Dark-red classification is based on HSV value (V), not hue alone.
# Tune these against real arena samples; camera exposure changes these values.
DARK_RED_MAX_VALUE = 125
DARK_RED_MIN_SATURATION = 110
RED_MIN_SOLIDITY = 0.45
RED_MIN_ASPECT_RATIO = 0.20
RED_MAX_ASPECT_RATIO = 5.0
RED_CONFIRM_FRAMES = 8
RED_CONFIRM_IOU = 0.20
RED_TRACK_MAX_MISSES = 5

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

        # Red candidate segmentation only. This is NOT a danger-zone decision.
        red_mask_1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        red_mask_2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Reject weakly saturated/dim red-like pixels. This reduces some
        # brown/skin false positives but cannot semantically identify a marker.
        red_mask = cv2.bitwise_and(
            red_mask,
            cv2.inRange(
                hsv,
                np.array([0, RED_MIN_SATURATION, RED_MIN_VALUE]),
                np.array([180, 255, 255]),
            ),
        )

        # Dark-red mask: retain only red-hue pixels whose HSV value is low enough.
        # This is a heuristic, not a semantic guarantee: shadows can darken skin
        # and bright red markers can be missed if overexposed.
        dark_red_mask = cv2.bitwise_and(
            red_mask,
            cv2.inRange(
                hsv,
                np.array([0, DARK_RED_MIN_SATURATION, RED_MIN_VALUE]),
                np.array([180, 255, DARK_RED_MAX_VALUE]),
            ),
        )

        # Reduce small noise in both masks.
        green_mask = cv2.erode(green_mask, kernel, iterations=1)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        # Colour-coded mask window.
        combined_mask = np.zeros_like(frame)
        combined_mask[green_mask > 0] = OBJECT_COLORS["GREEN"]
        combined_mask[red_mask > 0] = (0, 165, 255)  # orange = all red candidates
        combined_mask[dark_red_mask > 0] = (0, 0, 255)  # red = dark-red subset

        # Green remains colour-segmented. Red regions are candidates only.
        detections = {"GREEN": green_mask, "DARK RED DANGER CANDIDATE": dark_red_mask}
        hud_info = {}

        for color_name, mask in detections.items():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            draw_color = OBJECT_COLORS["GREEN"] if color_name == "GREEN" else (0, 0, 255)
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

                # Geometry checks are diagnostic only; they do not confirm danger.
                perimeter = cv2.arcLength(contour, True)
                solidity = area / max(1.0, float(w * h))
                aspect_ratio = w / max(1.0, float(h))
                shape_ok = (
                    RED_MIN_SOLIDITY <= solidity <= 1.0
                    and RED_MIN_ASPECT_RATIO <= aspect_ratio <= RED_MAX_ASPECT_RATIO
                ) if color_name == "RED CANDIDATE" else True

                if color_name == "GREEN":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                    label = f"GREEN #{object_index}"
                else:
                    # Orange outline explicitly communicates unconfirmed candidate.
                    cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                    label = f"DARK RED CANDIDATE #{object_index}"
                    cv2.putText(
                        frame,
                        "DARK RED CANDIDATE - VERIFY",
                        (x, min(frame_height - 8, y + h + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, draw_color, 1, cv2.LINE_AA
                    )

                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(
                    frame, label, (x, max(18, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 2, cv2.LINE_AA
                )

                objects.append({
                    "index": object_index,
                    "centre": (cx, cy),
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "area": int(area),
                    "solidity": solidity,
                    "aspect_ratio": aspect_ratio,
                    "shape_check": shape_ok,
                })

            hud_info[color_name] = objects

        # HUD shows dark-red candidates only; this is not yet a confirmed restricted-zone classifier.
        hud_y = 25
        for color_name in ("GREEN", "DARK RED DANGER CANDIDATE"):
            draw_color = OBJECT_COLORS["GREEN"] if color_name == "GREEN" else (0, 0, 255)
            objects = hud_info[color_name]
            lines = [f"{color_name}: {len(objects)}"]
            for info in objects[:3]:
                lines.append(
                    f"#{info['index']} C{info['centre']} "
                    f"XY({info['offset_x']:+d},{info['offset_y']:+d}) "
                    f"A{info['area']}px2"
                )
                if color_name == "DARK RED DANGER CANDIDATE":
                    lines.append(
                        f"  shape-check={info['shape_check']} "
                        f"solidity={info['solidity']:.2f}"
                    )
            if len(objects) > 3:
                lines.append(f"... +{len(objects) - 3} more")
            for line in lines:
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
                )
                cv2.rectangle(
                    frame, (8, hud_y - text_h - 3),
                    (16 + text_w, hud_y + baseline), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, line, (12, hud_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, draw_color, 1, cv2.LINE_AA
                )
                hud_y += 19
            hud_y += 4

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
