import glob
import cv2
import numpy as np
from CameraCalibration.chessBoard import ChessBoard


def write_intrinsic_matrix(camera_matrix, distortion_coefficient, mode: str):
    '''
    Write the camera intrinsic matrix and distortion coefficients to a file.

    Args:
        camera_matrix (numpy.ndarray): The camera matrix (3x3).
        distortion_coefficient (numpy.ndarray): The distortion coefficients.
        mode (str): The calibration mode ('normal' or 'fisheye').
    '''
    if mode == 'normal':
        with open('results/intrinsicNormal.py', mode='w', encoding='utf-8') as f:
            f.write('import numpy as np\n')
            f.write('camera_matrix = np.float32(' + str(camera_matrix.tolist()) + ')\n')
            f.write('distortion_coefficient = np.float32(' + str(distortion_coefficient.tolist()) + ')\n')
        print('Matrix has been written to intrinsicNormal.py')
    elif mode == 'fisheye':
        with open('results/intrinsicFisheye.py', mode='w', encoding='utf-8') as f:
            f.write('import numpy as np\n')
            f.write('camera_matrix = np.float32(' + str(camera_matrix.tolist()) + ')\n')
            f.write('distortion_coefficient = np.float32(' + str(distortion_coefficient.tolist()) + ')\n')
        print('Matrix has been written to intrinsicFisheye.py')


def calibrate(images_folder: str, cb: ChessBoard, mode='normal'):
    '''
    Calibrate the camera using images of a chessboard pattern.

    Args:
        images_folder (str): The folder containing the calibration images.
        cb (ChessBoard): The chessboard configuration.
        mode (str): The calibration mode ('normal' or 'fisheye').

    Returns:
        tuple: The camera matrix and distortion coefficients.
    '''
    image_points = []  # 2D corner points in the image
    world_points = []  # 3D points in the world coordinate system

    # Generate 3D points for the chessboard corners
    wps = np.zeros((cb.row * cb.col, 1, 3), np.float32)
    wps[:, 0, :2] = np.mgrid[0:cb.col, 0:cb.row].T.reshape(-1, 2)
    wps = wps * cb.width  # Convert from pixels to millimeters

    h, w = 0, 0  # Image height and width

    # Load all calibration images
    image_sets = glob.glob(images_folder + '/*.png')
    for image_path in image_sets:
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Find chessboard corners
        _, corners = cv2.findChessboardCorners(gray, (cb.col, cb.row), None)
        if corners is not None:
            image_points.append(corners)
            world_points.append(wps)

        # Check for mismatched points
        if len(world_points[0]) != len(image_points[0]):
            print(f'No match in {image_path}!')
            return

    # Perform camera calibration based on the selected mode
    if mode == 'normal':
        _, camera_matrix, distortion_coefficient, rvec, tvec = cv2.calibrateCamera(
            world_points, image_points, (w, h), None, None)

        print('Calibration finished!')
        print('Number of boards:', len(rvec), len(tvec))
        print('K:', camera_matrix)
        print('D:', distortion_coefficient)

        write_intrinsic_matrix(camera_matrix, distortion_coefficient, mode)
        return camera_matrix, distortion_coefficient

    elif mode == 'fisheye':
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(len(image_sets))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(len(image_sets))]
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            world_points, image_points, (w, h), K, D, rvecs, tvecs, criteria=criteria)

        print('Calibration finished!')
        print('K:', K)
        print('D:', D)

        write_intrinsic_matrix(K, D, mode)
        return K, D


if __name__ == '__main__':
    # Initialize chessboard configuration
    cb = ChessBoard(9, 6, 10)  # Columns, rows, and width (in mm)

    # Select calibration mode
    mode = 'normal'  # or 'fisheye'

    # Perform calibration
    calibrate('./data', cb, mode)