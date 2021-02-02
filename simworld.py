import copy

class Board:
  """
  A Board instance permits a single gameplay episode and can only be played through once
  During training, the simulation environment will thus intantiate new boards iteratively

  TODO: Er det kanskje lurere å ha en reset funksjon inne i board som tillater
  å bruke samme brettet og sparer litt utregningstid? 
  """
  
  def __init__(self, size, open_cells, track_history = False):
    """ 
    Initializes a board instance with subclass logic
    List representation of board is stored as current_state
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
    s = 0 
    for line in self.current_state:
      s += sum(line)
    return s

  def is_won(self):
    """
    Checks if current board state is a winning state (=only one piece left)
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
    where action[0] is current piece cell and action[1] is target cell
    :param action: two cell coordinates
    """
    if self.is_legal_action(action):
      # Change current coord from 1 to 0 and target coordinate from 0 to 1
      piece_row, piece_col = action[0][0], action[0][1]
      target_row, target_col = action[1][0], action[1][1]
      
      self.current_state[piece_row][piece_col] = 0
      self.current_state[target_row][target_col] = 1

      # Remove piece in the middle that was overjumped
      mid_row = int((piece_row+target_row)/2)
      mid_col = int((piece_col+target_col)/2)
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

    # Check if cells are within board
    if not self.validate_cell_coordinates(action):
      return False
    
    piece_row, piece_col = action[0][0], action[0][1]
    target_row, target_col = action[1][0], action[1][1]
    # Check if current cell has piece and target cell is empty
    if self.current_state[piece_row][piece_col] != 1 or self.current_state[target_row][target_col] != 0:
      return False

    row_shift, col_shift = target_row - piece_row, target_col - piece_col 

    # Check that there is actually a piece in the cell to jump over
    middle_row, middle_col = (piece_row + (row_shift/2), piece_col + (col_shift/2))
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
        print("--"*10)
        print(f"Action made: {action} \nResulted in:")
        self.show(board)  
    else:
      print("Track history was turned off, no gameplay history available")

  def validate_cell_coordinates(self, action):
    """
    Checks whether cell coordinates involved in an action are both within board
    Logic depends on board type, see method defined in subclass
    :param action: tuple containing piece cell and target cell
    """
    pass

  def get_legal_actions(self):
    """
    Returns all legal actions permissible in the current board state
    """
    pass

class DiamondBoard(Board):
  """
  Diamond-shaped board.
  """

  def __init__(self, size, open_cells, track_history = False):
    """ 
    Constructs board as defined in https://www.idi.ntnu.no/emner/it3105/materials/hex-board-games.pdf
    :param size: input param given by user, interpretation depends on type
    :param open_cells: Coordinates of cells to be initialized as empty
    :param track_history: Facilitates tracking of steps within an entire game for visualization purposes
    
    
    Board indices [row,col] works as follows:
     row + 1 -> next row below
     col + 1 -> next element horizontally to the right
    
    Example of diamond board of size 3 illustrated below

    [[ (0,0) , (0,1) , (0,2)  ]
     [ (1,0) , (1,1) , (1,2)  ]
     [ (2,0) , (2,1) , (2,2)  ]]

    """
    super().__init__(size, open_cells, track_history)
  
    # Builds board and removes open cells

    for i in range(size):
      self.current_state.append([1]*size)
    # Change open cells to 0 in board
    for cell in open_cells:
      if not self.validate_cell_coordinates([cell]):
        raise Exception(f"Cell input {cell} was outside of board")
      row, col = cell[0], cell[1]
      self.current_state[row][col] = 0
    
    # Append None action an snapshot of initial board state to history if tracking is ON
    if self.track_history:
      self.history.append((None, copy.deepcopy(self.current_state)))
    
  def validate_cell_coordinates(self, action):
    for cell in action:
      for x in cell:
        if x < 0 or x > self.size-1: # Checks whether index is within board
          return False
    return True
    

  def get_legal_actions(self):
    """
    Will calculate number of pieces remaining and search through legal actions
    by looping over empty cells or filled cells depending on which number is the
    smallest
    """
    legal_actions = []
    pieces_left = self.count_pieces()
    total_cells = self.size ** 2
    empty_cells = total_cells - pieces_left

    # Search through empty cells if the board has fewer empty cells, else search trough filled cells
    search_type = 0 if empty_cells < pieces_left else 1

    for row in range(self.size):
      for col in range(self.size):
        if self.current_state[row][col] == search_type:
          potential_cells = [(row, col-2), (row, col+2),      # Horizontal
                             (row-2, col), (row+2, col),      # Vertical
                             (row+2, col-2), (row-2, col+2)]   # Diagonal
                                  
          # Filter out cells that would consitute illegal actions
          for cell in potential_cells:
            potential_action = [cell, (row,col)] if search_type == 0 else [(row,col), cell]
            if self.is_legal_action(potential_action):
              legal_actions.append(potential_action)
    
    return legal_actions

class TriangleBoard(Board):

  def __init__(self, size, open_cells, track_history = False):
    """ 
    Constructs board as defined in https://www.idi.ntnu.no/emner/it3105/materials/hex-board-games.pdf
    :param size: input param given by user, interpretation depends on type
    :param open_cells: Coordinates of cells to be initialized as empty
    :param track_history: Facilitates tracking of steps within an entire game for visualization purposes
    
    
    Board indices [row,col] works as follows:
     row + 1 -> next row below
     col + 1 -> next element horizontally to the right
    
    Example of Triangular board of size 3 illustrated below

    [[ (0,0) ]
     [ (1,0) , (1,1) ]
     [ (2,0) , (2,1) , (2,2)  ]]

    """
    super().__init__(size, open_cells, track_history)

    # Builds board with 1's
    for i in range(size):
      self.current_state.append([1]*(i+1))
    # Change open cells to 0 in board
    for cell in open_cells:
      if not self.validate_cell_coordinates([cell]):
        raise Exception(f"Cell input {cell} was outside of board")
      row, col = cell[0], cell[1]
      self.current_state[row][col] = 0
    
    # Append None action and snapshot of initial board state to history if tracking is ON
    if self.track_history:
      self.history.append((None, copy.deepcopy(self.current_state)))
    
  def validate_cell_coordinates(self, action):
    for cell in action:
      row, col = cell[0], cell[1]
      if row < 0 or row > self.size-1 or col < 0 or col > len(self.current_state[row])-1 :
        return False
    return True


  def get_legal_actions(self):
    """
    Will calculate number of pieces remaining and search through legal actions
    by looping over empty cells or filled cells depending on which number is the
    smallest
    """
    legal_actions = []
    pieces_left = self.count_pieces()
    total_cells = (self.size * self.size+1) / 2 # Sum of 1..n = (n*(n+1))/2
    empty_cells = total_cells - pieces_left

    # Search through empty cells if the board has fewer empty cells, else search trough filled cells
    search_type = 0 if empty_cells < pieces_left else 1

    for row in range(self.size):
      for col in range(len(self.current_state[row])):
        if self.current_state[row][col] == search_type:
          potential_cells = [(row, col-2), (row, col+2),      # Horizontal
                             (row-2, col), (row+2, col),      # Vertical
                             (row+2, col+2), (row-2, col-2)]   # Diagonal
          
          # Filter out cells that would consitute illegal actions
          for cell in potential_cells:
            potential_action = [cell, (row,col)] if search_type == 0 else [(row,col), cell]
            if self.is_legal_action(potential_action):
              legal_actions.append(potential_action)
    return legal_actions

class Environment:
  """
  Environment that allows several plays of a board game and keeps track of reward parameters
  and can vary certain other parameters between games if desired
  """

  def __init__(self, step_reward, final_reward, loser_penalty, boardsize, open_cells, board_type = "Diamond", track_history = False):
    """
    Initializes a gaming environment. 
    :param step_reward: reward for making a single move in the game
    :param final_reward: Treated as a multiple of total accumulatable step_reward if step_reward != 0 otherwise a single number
    :param loser_penalty: Sets penalty (negative reward) for ending up in final state without winning
    :param boardsize: Decides size of board to be played inside environment
    :param open_cells: A list of cells to initialize as empty in the boards
    :param track_history: Whether to track gameplay within boards for visualization later
    """

    self.step_reward = step_reward
    
    if board_type not in {"Diamond", "Triangle"}:
      raise Exception(f"Board type must be either 'Diamond' or 'Traingle', input was {board_type}")
    
    self.Board_class = DiamondBoard if board_type == "Diamond" else TriangleBoard
    
    self.board = self.Board_class(boardsize, open_cells, track_history)
    
    self.final_reward = final_reward*step_reward*self.board.count_pieces() if step_reward != 0 else final_reward
    self.loser_penalty = loser_penalty

  def get_state(self):
    """
    Returns current state of boardgame
    """
    return self.board.get_state()

  def get_actions(self):
    """
    Returns possible actions from current state of boardgame
    """
    actions = self.board.get_legal_actions()
    return actions if len(actions) > 0 else None

  def perform_action(self, action):
    """
    Performs an action on the current boardgame and returns the reward it yielded
    """
    self.board.update(action)
    if self.board.is_lost():
      return self.loser_penalty
    if self.board.is_won():
      return self.final_reward
    else:
      return self.step_reward
    
    
  def game_is_finished(self):
    """
    Checks whether current boardgame is in a final state (either won or lost)
    """
    return self.board.is_finished()

  def new_game(self):
    """
    Resets boardgame and starts a new one
    """
    self.board = self.Board_class(self.board.size, self.board.open_cells, self.board.track_history)
  


#import random
#from matplotlib import pyplot as plt
  

# Regarding Size 4 Diamond board: the only two of the four middle cells that yield a solvable board are:
# env = Environment(1,1, loser_penalty=-5,boardsize=4,open_cells=[(1,2)], board_type="Diamond", track_history=True)
#env = Environment(1,1, loser_penalty=-5,boardsize=4,open_cells=[(2,1)], board_type="Diamond", track_history=True)

#env = Environment(1,1, loser_penalty=0,boardsize=4,open_cells=[(5,1)], board_type="Diamond", track_history=True)



#scores = []
#x = []
#i = 0
#while not env.board.is_won():
#  env.new_game()
#  score = 0
#  i +=1
#  if i % 50 == 0:
#    print(f"playing game number {i}")
#  while not env.game_is_finished():
#    a = random.choice(env.get_actions())
#    score += env.perform_action(a)
#  scores.append(score)
#  x.append(i)

#plt.plot(x, scores)

#env.board.show_gameplay()

#plt.plot(x, scores)

#env.board.show_gameplay()
