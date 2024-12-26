import cv2
import numpy as np


# Define image points and world points
image_points = [(397, 538), (292, 521), (308, 433), (379, 440), (372, 391), (458, 449), (512, 555)]
world_points = [(0, 600), (-300, 600), (-300, 900), (0, 900), (0, 1200), (300, 900), (300, 600)]

# Convert lists to numpy arrays
image_points = np.array(image_points, dtype=np.float32)
world_points = np.array(world_points, dtype=np.float32)

# Open the video file
cap = cv2.VideoCapture('data/OriginalVideo.avi')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data/ProcessedVideo.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Compute the homography matrix
    H, _ = cv2.findHomography(image_points, world_points)

    # Apply the homography matrix (e.g., map image points to world points)
    # Add specific mapping logic here as needed
    # Example: Map a point in the image to the world plane
    point_in_image = np.array([[397, 538]], dtype=np.float32)  # Point in the image
    point_in_image = point_in_image.reshape(-1, 1, 2)
    point_in_world = cv2.perspectiveTransform(point_in_image, H)  # Map to the world plane
    point_in_world = point_in_world.reshape(-1, 2)

    # Annotate the mapped point on the image
    x_world, y_world = point_in_world[0]
    cv2.circle(frame, (int(point_in_image[0][0][0]), int(point_in_image[0][0][1])), 5, (0, 255, 0), -1)  # Image point
    cv2.putText(frame, f'World point: ({x_world:.2f}, {y_world:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the current frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()