import cv2
import numpy as np
import math
import time
from collections import deque

# HSV ranges
LOWER_GREEN = np.array([35, 80, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Red wraps around the HSV hue scale. Tighter hue/saturation thresholds help reject brown/skin tones.
LOWER_RED_1 = np.array([0, 150, 80])
UPPER_RED_1 = np.array([7, 255, 255])
LOWER_RED_2 = np.array([173, 150, 80])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 500

# Tracking settings (no centre-history trail is drawn).
TRAIL_LENGTH = 150
MOVEMENT_THRESHOLD = 5
STATIONARY_TIME = 0.7

# Raspberry Pi Camera Module / Raspberry Pi AI Camera (IMX500)
# Requires Picamera2 installed on the Raspberry Pi OS system Python.
try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: Picamera2 is not installed. Install it on Raspberry Pi OS with:")
    print("sudo apt update && sudo apt install -y python3-picamera2")
    raise SystemExit(1)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(2.0)

print("Picamera2 started. Waiting for frames...")
frame = None
for i in range(10):
    try:
        frame_rgb = picam2.capture_array()
        if frame_rgb is not None and frame_rgb.size > 0:
            # Convert the Picamera2 RGB frame to OpenCV's BGR convention.
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            print("First valid frame received!")
            break
    except Exception as exc:
        print(f"Frame {i + 1} capture failed: {exc}")
    time.sleep(0.3)

if frame is None:
    print("ERROR: No frames received after 10 attempts.")
    picam2.stop()
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
        try:
            frame_rgb = picam2.capture_array()
            if frame_rgb is None or frame_rgb.size == 0:
                raise RuntimeError("Empty frame received")
            # Picamera2 supplies RGB888; OpenCV processing below uses BGR.
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            consecutive_read_failures += 1
            print(
                f"WARNING: Could not capture frame "
                f"({consecutive_read_failures}/10): {exc}"
            )
            if consecutive_read_failures >= 10:
                print("ERROR: Camera stream failed repeatedly; stopping.")
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

        for color_name, mask in detections.items():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            state = tracking[color_name]
            draw_color = OBJECT_COLORS[color_name]

            valid_contours = [
                contour for contour in contours
                if cv2.contourArea(contour) > MIN_CONTOUR_AREA
            ]

            if not valid_contours:
                state["previous_cx"] = None
                state["previous_cy"] = None
                state["history"].clear()
                state["stationary"] = False
                continue

            # Preserve original behaviour: track the largest valid object of each colour.
            contour = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Pixel offsets from the camera image centre.
            offset_x = cx - frame_center_x
            offset_y = cy - frame_center_y

            if state["previous_cx"] is not None:
                movement = math.sqrt(
                    (cx - state["previous_cx"]) ** 2
                    + (cy - state["previous_cy"]) ** 2
                )
                if movement > MOVEMENT_THRESHOLD:
                    state["last_movement_time"] = time.time()
                    state["stationary"] = False
                elif time.time() - state["last_movement_time"] > STATIONARY_TIME:
                    state["stationary"] = True

            state["previous_cx"], state["previous_cy"] = cx, cy

            # Draw object boundary and centre marker.
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), draw_color, 2)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), draw_color, 2)

            # A small red danger zone when a red region is detected.
            # This is a visual alert margin, not a real-world distance estimate.
            if color_name == "RED":
                danger_padding = 18
                dx1 = max(0, x - danger_padding)
                dy1 = max(0, y - danger_padding)
                dx2 = min(frame_width - 1, x + w + danger_padding)
                dy2 = min(frame_height - 1, y + h + danger_padding)

                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (dx1, dy1), (dx2, dy2), (0, 0, 255), thickness=-1
                )
                cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                cv2.putText(
                    frame, "DANGER ZONE", (dx1, max(18, dy1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                    cv2.LINE_AA
                )

            status = "STATIONARY" if state["stationary"] else "MOVING"
            hud_info[color_name] = {
                "centre": (cx, cy),
                "offset_x": offset_x,
                "offset_y": offset_y,
                "area": int(area),
                "status": status,
            }

        # Fixed top-left HUD for both colours.
        hud_y = 28
        for color_name in ("GREEN", "RED"):
            draw_color = OBJECT_COLORS[color_name]
            if color_name in hud_info:
                info = hud_info[color_name]
                lines = [
                    f"{color_name}: DETECTED",
                    f"Centre: ({info['centre'][0]}, {info['centre'][1]})",
                    f"X offset: {info['offset_x']:+d} px",
                    f"Y offset: {info['offset_y']:+d} px",
                    f"Area: {info['area']} px^2",
                    f"Status: {info['status']}",
                ]
            else:
                lines = [f"{color_name}: NOT DETECTED"]

            for line in lines:
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
                )
                cv2.rectangle(
                    frame, (10, hud_y - text_h - 4),
                    (20 + text_w, hud_y + baseline), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, line, (15, hud_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, draw_color, 2, cv2.LINE_AA
                )
                hud_y += 23
            hud_y += 8

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
    picam2.stop()
    cv2.destroyAllWindows()
    print("Picamera2 stopped.")
