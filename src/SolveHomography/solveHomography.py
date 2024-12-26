import cv2
import numpy as np


is_check = False  # Set to True if you want to check the points you have set


def solve_homography():
    '''
    Solve the homography matrix using predefined image and world points.
    The result is saved to a file.
    '''
    # 2D points in the image plane
    image_points = [(397, 538), (292, 521), (308, 433), (379, 440), (372, 391), (458, 449), (512, 555)]

    # 3D points in the real world space (ground plane z=0)
    world_points = [(0, 600), (-300, 600), (-300, 900), (0, 900), (0, 1200), (300, 900), (300, 600)]

    # Convert lists to numpy arrays
    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)

    # Compute the homography matrix
    H, _ = cv2.findHomography(image_points, world_points)
    cv2.waitKey(0)

    # Save the homography matrix to a file
    with open('result.py', mode='w', encoding='utf-8') as f:
        f.write('import numpy as np\n')
        f.write('homography_matrix = np.float32(' + str(H.tolist()) + ')')

    print('Homography matrix has been written to result.py')


def click_corner(event, x, y, flags, param):
    '''
    Mouse callback function to mark and display clicked points on the image.

    Args:
        event (int): The type of mouse event.
        x (int): The x-coordinate of the mouse click.
        y (int): The y-coordinate of the mouse click.
        flags (int): Additional flags.
        param: Additional parameters.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = f'{x},{y}'
        cv2.circle(image, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)


if __name__ == '__main__':
    file = 'data/DedistortedImage.png'  # The image to use

    if not is_check:
        solve_homography()
    else:
        # Check the points by visualizing them on the image
        image = cv2.imread(file)
        cv2.destroyAllWindows()

        cv2.namedWindow('solveHomography')
        cv2.setMouseCallback('solveHomography', click_corner)

        while True:
            cv2.imshow('solveHomography', image)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or key == ord('Q'):
                break