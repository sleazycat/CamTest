import cv2
import time
import math
from ultralytics import YOLO


# ============================================================
# SQUADRONE
# AEROTHON 2026 - MISSION 2
# AUTONOMOUS CORRIDOR OBSTACLE PERCEPTION
#
# MAC TEST VERSION
#
# Camera:
#   Camera 1 = working iPhone camera
#
# IMPORTANT:
#   This script does NOT control the drone.
#   It only analyses the camera image.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 1

MODEL_NAME = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.25


# ------------------------------------------------------------
# Flight corridor representation
# ------------------------------------------------------------

# This is NOT the physical 3.5 m corridor.
#
# It represents the portion of the camera image that corresponds
# to the drone's immediate forward flight path.

FLIGHT_CORRIDOR_WIDTH = 0.45
FLIGHT_CORRIDOR_HEIGHT = 0.55


# ------------------------------------------------------------
# Risk thresholds
# ------------------------------------------------------------

CAUTION_RISK = 30
DANGER_RISK = 60
EMERGENCY_RISK = 85


# ------------------------------------------------------------
# Apparent size thresholds
# ------------------------------------------------------------

SMALL_OBJECT = 0.05
MEDIUM_OBJECT = 0.15
LARGE_OBJECT = 0.30


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("==============================================")
print(" SQUADRONE - AEROTHON MISSION 2")
print(" CORRIDOR OBSTACLE PERCEPTION")
print("==============================================")
print()

print("Loading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded.")
print()


# ============================================================
# CAMERA
# ============================================================

print("Opening Camera 1...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open Camera 1."
    )


# Camera warm-up

time.sleep(1)

ret = False
frame = None

for _ in range(20):

    ret, frame = cap.read()

    if ret:
        break

    time.sleep(0.1)


if not ret:

    cap.release()

    raise RuntimeError(
        "Camera opened but no frames were received."
    )


print(
    f"Camera working: "
    f"{frame.shape[1]} x {frame.shape[0]}"
)

print()
print("Press Q to quit.")
print()


# ============================================================
# TRACKING MEMORY
# ============================================================

previous_objects = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# ------------------------------------------------------------


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


# ------------------------------------------------------------


def box_area(box):

    x1, y1, x2, y2 = box

    return max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )


# ------------------------------------------------------------


def box_overlaps_corridor(
    box,
    corridor
):

    x1, y1, x2, y2 = box

    cx1, cy1, cx2, cy2 = corridor

    overlap_x = (
        x1 < cx2 and
        x2 > cx1
    )

    overlap_y = (
        y1 < cy2 and
        y2 > cy1
    )

    return (
        overlap_x and
        overlap_y
    )


# ------------------------------------------------------------


def calculate_overlap_ratio(
    box,
    corridor
):

    x1, y1, x2, y2 = box

    cx1, cy1, cx2, cy2 = corridor

    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)

    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)

    width = max(
        0,
        ix2 - ix1
    )

    height = max(
        0,
        iy2 - iy1
    )

    intersection = width * height

    object_area = box_area(box)

    if object_area <= 0:
        return 0

    return intersection / object_area


# ------------------------------------------------------------


def get_position(
    cx,
    cy,
    frame_width,
    frame_height
):

    centre_x = frame_width / 2
    centre_y = frame_height / 2

    dx = cx - centre_x
    dy = cy - centre_y

    horizontal_limit = frame_width * 0.15
    vertical_limit = frame_height * 0.15

    if dx < -horizontal_limit:

        horizontal = "LEFT"

    elif dx > horizontal_limit:

        horizontal = "RIGHT"

    else:

        horizontal = "CENTER"


    if dy < -vertical_limit:

        vertical = "UP"

    elif dy > vertical_limit:

        vertical = "DOWN"

    else:

        vertical = "CENTER"


    return horizontal, vertical


# ------------------------------------------------------------


def calculate_risk(
    confidence,
    size_ratio,
    overlap_ratio,
    approach_rate
):

    risk = 0


    # Detection confidence

    risk += confidence * 20


    # Object size

    if size_ratio >= LARGE_OBJECT:

        risk += 35

    elif size_ratio >= MEDIUM_OBJECT:

        risk += 25

    elif size_ratio >= SMALL_OBJECT:

        risk += 10


    # How much of the object lies inside
    # the flight corridor.

    risk += overlap_ratio * 30


    # Apparent approach

    if approach_rate is not None:

        if approach_rate > 0.30:

            risk += 30

        elif approach_rate > 0.15:

            risk += 20

        elif approach_rate > 0.05:

            risk += 10


    return min(
        100,
        risk
    )


# ------------------------------------------------------------


def risk_state(risk):

    if risk >= EMERGENCY_RISK:

        return "EMERGENCY"

    if risk >= DANGER_RISK:

        return "DANGER"

    if risk >= CAUTION_RISK:

        return "CAUTION"

    return "SAFE"


# ------------------------------------------------------------


def avoidance_direction(
    horizontal,
    vertical
):

    if horizontal == "LEFT":

        return "RIGHT"

    if horizontal == "RIGHT":

        return "LEFT"

    if vertical == "UP":

        return "DOWN"

    if vertical == "DOWN":

        return "UP"

    return "STOP"


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Failed to read camera frame."
        )

        continue


    frame_height, frame_width = frame.shape[:2]


    # ========================================================
    # IMAGE CENTRE
    # ========================================================

    centre_x = frame_width // 2
    centre_y = frame_height // 2


    # ========================================================
    # FLIGHT CORRIDOR
    # ========================================================

    corridor_width = int(
        frame_width *
        FLIGHT_CORRIDOR_WIDTH
    )

    corridor_height = int(
        frame_height *
        FLIGHT_CORRIDOR_HEIGHT
    )


    corridor_x1 = int(
        centre_x -
        corridor_width / 2
    )

    corridor_x2 = int(
        centre_x +
        corridor_width / 2
    )

    corridor_y1 = int(
        centre_y -
        corridor_height / 2
    )

    corridor_y2 = int(
        centre_y +
        corridor_height / 2
    )


    corridor = (
        corridor_x1,
        corridor_y1,
        corridor_x2,
        corridor_y2
    )


    # Draw flight corridor

    cv2.rectangle(
        frame,
        (
            corridor_x1,
            corridor_y1
        ),
        (
            corridor_x2,
            corridor_y2
        ),
        (255, 255, 255),
        2
    )


    # Centre crosshair

    cv2.drawMarker(
        frame,
        (
            centre_x,
            centre_y
        ),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        30,
        2
    )


    # ========================================================
    # YOLO
    # ========================================================

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    current_objects = []


    for result in results:

        for detection in result.boxes:

            confidence = float(
                detection.conf[0]
            )

            class_id = int(
                detection.cls[0]
            )

            class_name = model.names[
                class_id
            ]


            x1, y1, x2, y2 = map(
                int,
                detection.xyxy[0]
            )


            box = (
                x1,
                y1,
                x2,
                y2
            )


            cx, cy = box_center(
                box
            )


            area = box_area(
                box
            )


            image_area = (
                frame_width *
                frame_height
            )


            size_ratio = (
                area /
                image_area
            )


            # Is the detected object in the
            # drone's projected flight corridor?

            in_corridor = (
                box_overlaps_corridor(
                    box,
                    corridor
                )
            )


            overlap_ratio = (
                calculate_overlap_ratio(
                    box,
                    corridor
                )
            )


            horizontal, vertical = (
                get_position(
                    cx,
                    cy,
                    frame_width,
                    frame_height
                )
            )


            # =================================================
            # TRACK OBJECT
            # =================================================

            previous_area = None

            best_previous_iou = 0

            for previous in previous_objects.values():

                if previous["class"] != class_name:

                    continue

                iou = calculate_iou(
                    box,
                    previous["box"]
                )

                if iou > best_previous_iou:

                    best_previous_iou = iou

                    previous_area = (
                        previous["area"]
                    )


            approach_rate = None


            if (
                previous_area is not None
                and
                previous_area > 0
            ):

                approach_rate = (
                    area -
                    previous_area
                ) / previous_area


            # =================================================
            # RISK
            # =================================================

            risk = calculate_risk(
                confidence,
                size_ratio,
                overlap_ratio,
                approach_rate
            )


            # Objects outside the corridor are
            # lower priority.

            if not in_corridor:

                risk *= 0.35


            state = risk_state(
                risk
            )


            current_objects.append({

                "class":
                    class_name,

                "box":
                    box,

                "area":
                    area,

                "risk":
                    risk,

                "in_corridor":
                    in_corridor,

            })


            # =================================================
            # DRAW
            # =================================================

            if in_corridor:

                thickness = 4

            else:

                thickness = 2


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness
            )


            # Centre

            cv2.circle(
                frame,
                (
                    int(cx),
                    int(cy)
                ),
                5,
                (0, 0, 255),
                -1
            )


            # =================================================
            # TEXT
            # =================================================

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            status = (
                f"{state} "
                f"Risk:{risk:.0f}"
            )


            cv2.putText(
                frame,
                status,
                (
                    x1,
                    y2 + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


            if in_corridor:

                position = (
                    f"PATH: YES | "
                    f"{horizontal}-{vertical}"
                )

            else:

                position = "PATH: NO"


            cv2.putText(
                frame,
                position,
                (
                    x1,
                    y2 + 48
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )


    # ========================================================
    # FIND MOST DANGEROUS OBJECT
    # ========================================================

    highest_risk = 0

    dangerous_object = "NONE"

    dangerous_position = "CENTER"

    dangerous_vertical = "CENTER"


    for obj in current_objects:

        if obj["risk"] > highest_risk:

            highest_risk = obj["risk"]

            dangerous_object = obj["class"]

            cx, cy = box_center(
                obj["box"]
            )

            (
                dangerous_position,
                dangerous_vertical
            ) = get_position(
                cx,
                cy,
                frame_width,
                frame_height
            )


    global_state = risk_state(
        highest_risk
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if highest_risk >= DANGER_RISK:

        command = avoidance_direction(
            dangerous_position,
            dangerous_vertical
        )

    elif highest_risk >= CAUTION_RISK:

        command = "SLOW / ANALYSE"

    else:

        command = "FORWARD"


    # ========================================================
    # HUD
    # ========================================================

    cv2.putText(
        frame,
        "AEROTHON 2026 | MISSION 2",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"COLLISION STATE: {global_state}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RISK: {highest_risk:.0f}/100",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"PRIMARY OBSTACLE: {dangerous_object}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RECOMMENDATION: {command}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE CURRENT OBJECTS FOR NEXT FRAME
    # ========================================================

    previous_objects = {}

    for index, obj in enumerate(
        current_objects
    ):

        previous_objects[index] = obj


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Squadrone - AeroTHON Corridor Perception",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Camera closed.")
print("Squadrone perception test stopped.")