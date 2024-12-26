class ChessBoard:
    '''
    A class to represent the chessboard used for camera calibration.

    Attributes:
        col (int): The number of corners along the columns (the odd side).
        row (int): The number of corners along the rows (the even side).
        width (int): The physical distance (in millimeters) between corners in the real world.
    '''
    def __init__(self, col, row, width) -> None:
        '''
        Initialize the ChessBoard with the given dimensions.

        Args:
            col (int): The number of corners along the columns.
            row (int): The number of corners along the rows.
            width (int): The physical distance between corners in millimeters.
        '''
        self.col = col  # Number of corners along the columns (the odd side)
        self.row = row  # Number of corners along the rows (the even side)
        self.width = width  # Physical distance between corners in millimeters