import cv2


def click_points(event, x, y, flags, param):
    '''
    Mouse callback function to capture clicked points on the image.

    Args:
        event (int): The type of mouse event.
        x (int): The x-coordinate of the mouse click.
        y (int): The y-coordinate of the mouse click.
        flags (int): Additional flags.
        param: Additional parameters.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Image Point: ({x}, {y})')
        image_points.append((x, y))


# Open the video file
video_path = 'data/OriginalVideo.avi'
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    raise ValueError(f'Failed to open video file: {video_path}')

# Create a window and bind the mouse callback function
cv2.namedWindow('Click Points')
cv2.setMouseCallback('Click Points', click_points)

# Initialize variables
image_points = []  # List to store the clicked points
frame_count = 0  # Frame counter

# Process the video frame by frame
while True:
    ret, frame = cap.read()  # Read the current frame
    if not ret:
        print('End of video or failed to read frame.')
        break

    frame_count += 1
    print(f'Processing frame {frame_count}')

    # Display the current frame
    cv2.imshow('Click Points', frame)

    # Wait for user input
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('n'):  # Press 'n' to skip the current frame
        continue

# Print all the clicked points
print('Image Points:', image_points)

# Release resources
cap.release()
cv2.destroyAllWindows()