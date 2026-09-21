"""
COCO 80-Class Registry and Utility Helpers for Advanced People & Object Analytics.
"""

COCO_CLASS_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "cellphone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

COCO_ID_TO_CLASS = {v: k for k, v in COCO_CLASS_TO_ID.items()}

# Default tracked classes in Advanced People & Object Analytics
DEFAULT_APA_TRACKED_CLASSES = [
    "person",
    "backpack",
    "handbag",
    "umbrella",
    "suitcase",
    "laptop",
    "mouse",
    "keyboard",
    "cell phone",
    "remote",
    "bottle",
    "cup",
    "book",
]


def resolve_class_ids(class_names: list[str] | None) -> list[int]:
    """
    Translates a list of class name strings into unique integer COCO class IDs.
    Defaults to DEFAULT_APA_TRACKED_CLASSES if None or empty.
    Always ensures class 0 (person) is included.
    """
    names = class_names if class_names else DEFAULT_APA_TRACKED_CLASSES
    ids = set()
    for name in names:
        clean_name = name.strip().lower()
        if clean_name in COCO_CLASS_TO_ID:
            ids.add(COCO_CLASS_TO_ID[clean_name])
    ids.add(0)  # Always include person
    return sorted(list(ids))


def calculate_bbox_iou(boxA: list[int | float], boxB: list[int | float]) -> float:
    """Calculates Intersection Over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def calculate_containment_ratio(object_box: list[int | float], person_box: list[int | float]) -> float:
    """
    Calculates fraction of object_box area that falls inside person_box.
    Returns value between 0.0 and 1.0.
    """
    xA = max(object_box[0], person_box[0])
    yA = max(object_box[1], person_box[1])
    xB = min(object_box[2], person_box[2])
    yB = min(object_box[3], person_box[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    obj_area = (object_box[2] - object_box[0]) * (object_box[3] - object_box[1])
    if obj_area <= 0:
        return 0.0
    return float(inter_area) / float(obj_area)
