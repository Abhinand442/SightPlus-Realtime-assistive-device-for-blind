from ultralytics import YOLO 
import cv2
import cvzone
import math



def detect_book_in_frame(frame):
    book_index = 73
    classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", " toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"]    

    model = YOLO("yolov8n.pt")

    results = model(frame, stream=True)
    detected_objects = []
    book_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            detected_objects.append(classNames[cls])

            if cls == book_index:
                book_detected = True 

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            cvzone.cornerRect(frame, bbox)

            # CONFIDENCE SCORE AND CLASS
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(frame, f"{classNames[cls]}  {conf}",
                               (max(0, x1), max(30, y1 - 20)),
                               scale=0.8, thickness=1)
    cv2.imshow("frame1", frame)

    return book_detected, detected_objects