import cv2
import numpy as np
import torch
from ultralytics import YOLO
from CalculateDistance.calculateDistance import calculate_distance
from CameraCalibration.Result.intrinsic_fisheye import camera_matrix, distortion_coefficient


# Load the YOLO model
model = YOLO('TrainingResults/weights/best.pt')

# Define the class names for the YOLO model
class_names = ['bump', 'speedbump']  # Class ID 0 is 'bump', class ID 1 is 'speedbump'

# Open the video captured by the fisheye camera
cap = cv2.VideoCapture('SolveHomography/data/OriginalVideo.avi')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object for saving the undistorted video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
undistorted_out = cv2.VideoWriter('UndistortedVideo.avi', fourcc, fps, (frame_width, frame_height))

# Create a VideoWriter object for saving the video with distance annotations
distance_out = cv2.VideoWriter('AnnotatedVideo.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the fisheye image
    undistorted_frame = cv2.fisheye.undistortImage(frame, camera_matrix, distortion_coefficient, Knew=camera_matrix)

    # Write the undistorted frame to the video
    undistorted_out.write(undistorted_frame)

    # Perform object detection
    with torch.no_grad():
        outputs = model(undistorted_frame)

    detection = outputs[0].boxes

    if detection.xyxy is not None and len(detection.xyxy) > 0:
        for box in detection:
            x, y, w, h = box.xywh[0]  # Get bounding box coordinates
            x1, y1 = x.item(), y.item() + h.item() / 2

            # Get class ID and confidence
            class_id = box.cls.item()
            confidence = box.conf.item()
            class_name = class_names[int(class_id)]  # Get class name from class ID

            # Calculate the distance
            distance = calculate_distance(x1, y1)

            # Add labels
            if 0 < distance < 10:
                # Draw the bounding box (red)
                cv2.rectangle(undistorted_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

                # Add class name and confidence label (yellow)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(undistorted_frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Add distance label (white)
                distance_label = f'Distance: {distance:.6f}'
                cv2.putText(undistorted_frame, distance_label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write the frame with distance annotations to the video
    distance_out.write(undistorted_frame)

# Release resources
cap.release()
undistorted_out.release()
distance_out.release()
cv2.destroyAllWindows()