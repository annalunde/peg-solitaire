import copy


class Board:
    """
    A Board instance permits a single gameplay episode and can only be played through once
    During training, the simulation environment will thus intantiate new boards iteratively

    NOTE: This class is not made to be initialized directly. It works as an interface/abstract class and requires
    implementation by subclasses like DiamondBoard to work.
    """

    def __init__(self, size, open_cells, track_history=False):
        """
        Initializes a board instance with subclass logic
        List representation of board is stored as current_state
        :param size: Size of board, see subclass implementation for details
        :param open_cells: Cells to be initialized as open
        :param track_history: Whether or not to keep track of boards previous states during play
        """
        self.size = size
        self.open_cells = open_cells
        self.current_state = []
        self.track_history = track_history
        self.history = [] if self.track_history else None

    def get_state(self):
        """
        Returns current state of the board in list format
        """
        return self.current_state

    def count_pieces(self):
        """
        Counts number of pieces left on board
        :return: int number of pieces left
        """
        s = 0
        for line in self.current_state:
            s += sum(line)
        return s

    def is_won(self):
        """
        Checks if current board state is a winning state (=only one piece left)
        :return: boolean
        """
        return self.count_pieces() == 1

    def is_lost(self):
        """
        Checks if current board state is a losing state (=no more actions and not won)
        """
        return self.is_finished() and not self.is_won()

    def is_finished(self):
        """
        Checks whether board game is finished (no more legal actions left)
        """
        return len(self.get_legal_actions()) == 0

    def update(self, action):
        """
        Moves a piece from action[0] to action[1]
        where action[0] is current piece cell coordinate and action[1] is target cell coordinate
        :param action: two cell coordinates
        """
        if self.is_legal_action(action):
            # Change current coord from 1 to 0 and target coordinate from 0 to 1
            piece_row, piece_col = action[0][0], action[0][1]
            target_row, target_col = action[1][0], action[1][1]

            self.current_state[piece_row][piece_col] = 0
            self.current_state[target_row][target_col] = 1

            # Remove piece in the middle that was overjumped
            mid_row = int((piece_row + target_row) / 2)
            mid_col = int((piece_col + target_col) / 2)
            self.current_state[mid_row][mid_col] = 0

        if self.track_history:
            # Record action made and snapshot of board state after update
            self.history.append((action, copy.deepcopy(self.current_state)))

    def is_legal_action(self, action):
        """
        Checks whether input action is a permitted move in the current board state
        :param action: two cell coordinates [(piece),(target)]
        :return: True or False
        """

        # Check that cells are within board
        if not self.validate_cell_coordinates(action):
            return False

        piece_row, piece_col = action[0][0], action[0][1]
        target_row, target_col = action[1][0], action[1][1]
        # Check that current cell has piece and target cell is empty
        if self.current_state[piece_row][piece_col] != 1 or self.current_state[target_row][target_col] != 0:
            return False

        row_shift, col_shift = target_row - piece_row, target_col - piece_col

        # Check that there is actually a piece in the cell to jump over
        middle_row, middle_col = (
            piece_row + (row_shift / 2), piece_col + (col_shift / 2))
        return self.current_state[int(middle_row)][int(middle_col)] == 1

    def show(self, b=None):
        """
        Displays the input board state b if b is provided, else current board state
        :param b: optional input board
        """
        b = b if b is not None else self.current_state
        for line in b:
            print(line)

    def show_gameplay(self):
        """
        Displays entire history of gameplay as action made -> resulting board
        """
        if self.track_history:
            for action, board in self.history:
                print("--" * 10)
                print(f"Action made: {action} \nResulted in:")
                self.show(board)
        else:
            print("Track history was turned off, no gameplay history available")

    def validate_cell_coordinates(self, action):
        """
        Checks whether both cell coordinates involved in the action are within board
        Logic depends on board type, see method defined in subclasses for details
        :param action: tuple containing piece cell and target cell [(piece), (target)]
        :return: True or False
        """
        pass

    def get_legal_actions(self):
        """
        Returns all actions that are permitted in the current board state
        """
        pass
