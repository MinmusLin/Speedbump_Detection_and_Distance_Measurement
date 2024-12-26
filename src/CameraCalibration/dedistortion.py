import cv2
import numpy as np


def dedistortion(image, camera_mode, size):
    '''
    Remove distortion from an image based on the camera mode.

    Args:
        image (numpy.ndarray): The distorted input image.
        camera_mode (str): The camera mode ('normal' or 'fisheye').
        size (tuple): The size of the image (height, width).

    Returns:
        numpy.ndarray: The undistorted image.
    '''
    if camera_mode == 'normal':
        from results.intrinsicNormal import camera_matrix, distortion_coefficient

        # Get the optimal new camera matrix
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion_coefficient, size, 0, size, centerPrincipalPoint=False)

        # Compute the undistortion maps
        mapX, mapY = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_coefficient, None, new_matrix, size, 5)

        # Apply the undistortion
        dedistorted_image = cv2.remap(image, mapX, mapY, cv2.INTER_LINEAR)

        print('Dedistortion finished!')
        cv2.imshow('Dedistorted Image', dedistorted_image)
        cv2.waitKey(0)

        return dedistorted_image

    elif camera_mode == 'fisheye':
        from results.intrinsicFisheye import camera_matrix, distortion_coefficient

        # Compute the undistortion maps for fisheye lens
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, distortion_coefficient, np.eye(3), camera_matrix,
            [int(i * 1.2) for i in size[::-1]], cv2.CV_16SC2)

        # Apply the undistortion
        dedistorted_image = cv2.remap(
            image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        print('Dedistortion finished!')
        cv2.imshow('Dedistorted Image', dedistorted_image)
        cv2.waitKey(0)

        return dedistorted_image


if __name__ == '__main__':
    # Path to the input image
    image_path = '../SolveHomography/data/OriginalImage.png'

    # Camera mode ('normal' or 'fisheye')
    camera_mode = 'fisheye'
    # camera_mode = 'normal'

    # Load the image
    image = cv2.imread(image_path)
    print(f'Image shape: {image.shape}')

    # Remove distortion from the image
    dedistorted_image = dedistortion(image, camera_mode, image.shape[:2])

    # Save the undistorted image
    cv2.imwrite('../SolveHomography/data/DedistortedImage.png', dedistorted_image)