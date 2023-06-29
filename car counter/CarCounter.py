# YOLOV8 - Car Counter
# Importing needed libraries
import numpy as np
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

# Load the video capture object to read the video file
cap = cv.VideoCapture("../Videos/cars.mp4")

# Load the YOLO model for object detection
model = YOLO('../yolo_weights/yolov8m.pt')

# Define a list of class names for the detected objects
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the mask image
mask = cv.imread('../Images/mask.png')

# Sort Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the limits of the counting area
limits = [400, 297, 673, 297]

# Initialize the list to store the IDs of counted objects
totalCounts = []

# Start processing the video frame by frame
while True:
    # Read the next frame from the video
    success, img = cap.read()
    if not success:
        break

    # Resize the mask to match the frame size
    mask = cv.resize(mask, (img.shape[1], img.shape[0]))
    mask = mask.astype(img.dtype)

    # Apply bitwise AND operation to the frame and the mask
    img_region = cv.bitwise_and(img, mask)

    # Perform object detection using the YOLO model
    results = model(img_region, stream=True)
    detections = np.empty((0, 5))

    # Process the detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate the width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Calculate the confidence score of the detected object
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get the class index of the detected object
            cls = int(box.cls[0])

            # Get the class name of the detected object
            currentClass = classNames[cls]

            # Check if the detected object belongs to a specific class and meets the confidence threshold
            if currentClass == "car" or currentClass == "motorbike" or currentClass == "bus" or currentClass == "truck" and conf > 0.3:
                # Create an array with the bounding box coordinates and confidence score
                currentArray = np.array([x1, y1, x2, y2, conf])
                # Append the current detection to the detections array
                detections = np.vstack((detections, currentArray))

    # Update the object tracker with the new detections
    results_tracker = tracker.update(detections)

    # Draw a line representing the counting area on the frame
    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Process the tracked objects
    for result in results_tracker:
        # Extract the bounding box coordinates and ID of the tracked object
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a rectangle around the tracked object using cvzone library
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))

        # Display the class name of the tracked object
        cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Draw a circle at the center of the tracked object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Check if the center of the tracked object is within the counting area
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # Check if the ID of the tracked object is already counted
            if totalCounts.count(id) == 0:
                # Add the ID to the list of counted objects
                totalCounts.append(id)
                # Draw a green line to indicate the object has been counted
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display the count of objects on the frame
    cvzone.putTextRect(img, f"Count: {len(totalCounts)}", (50, 50))

    # Show the processed frame
    cv.imshow("image", img)
    cv.waitKey(1)
