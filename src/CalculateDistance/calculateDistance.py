import numpy as np
from SolveHomography.result import homography_matrix


def calculate_distance(x, y):
    '''
    Calculate the distance of a point in the world coordinate system.

    Args:
        x (float): The x-coordinate of the point in the image.
        y (float): The y-coordinate of the point in the image.

    Returns:
        float: The calculated distance in meters.
    '''
    # Create a homogeneous coordinate from the input point
    homogeneous_coordinate = np.array([x, y, 1])

    # Transform the point to the world coordinate system using the homography matrix
    world_point = np.dot(homography_matrix, homogeneous_coordinate)

    # Normalize the world point by dividing by the third coordinate (homogeneous scaling)
    ratio = 1 / world_point[2]
    world_point *= ratio

    # Return the y-coordinate of the world point, converted from millimeters to meters
    return world_point[1] / 1000