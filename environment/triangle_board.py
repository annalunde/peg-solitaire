from matplotlib import pyplot as plt
import networkx as nx
import copy
from environment.board import Board


class TriangleBoard(Board):

    def __init__(self, size, open_cells, track_history=False):
        """
        Constructs board as defined in https://www.idi.ntnu.no/emner/it3105/materials/hex-board-games.pdf
        :param size: input param given by user, basically determines size of edges of the board.
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
            self.current_state.append([1] * (i + 1))
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
        """
        Checks whether both cell coordinates involved in the action are within board
        :param action: tuple containing piece cell and target cell [(piece), (target)]
        :return: True or False
        """
        for cell in action:
            row, col = cell[0], cell[1]
            if row < 0 or row > self.size - 1 or col < 0 or col > len(self.current_state[row]) - 1:
                return False
        return True

    def get_legal_actions(self):
        """
        Finds all actions that are permitted from the current state

        NOTE:
        For efficiency the function calculates number of pieces remaining and performs legal actions search
        by looping over empty cells or filled cells depending on which number is the smallest

        :return: List of actions [[(piece, (target)], [(piece, (target)]...]
        """
        legal_actions = []
        pieces_left = self.count_pieces()
        total_cells = (self.size * self.size + 1) / \
            2  # Sum of 1..n = (n*(n+1))/2
        empty_cells = total_cells - pieces_left

        # Search through empty cells if the board has fewer empty cells, else search trough filled cells
        search_type = 0 if empty_cells < pieces_left else 1

        for row in range(self.size):
            for col in range(len(self.current_state[row])):
                if self.current_state[row][col] == search_type:
                    # Potential cells correspond to neighbours in accordance with the board rules
                    potential_cells = [(row, col - 2), (row, col + 2),  # Horizontal moves
                                       # Vertical moves
                                       (row - 2, col), (row + 2, col),
                                       (row + 2, col + 2), (row - 2, col - 2)]  # Diagonal moves

                    # Filter out cells that would constitute illegal actions
                    for cell in potential_cells:
                        # Action depends on search type (jump to/jump from)
                        potential_action = [cell, (row, col)] if search_type == 0 else [
                            (row, col), cell]
                        if self.is_legal_action(potential_action):
                            legal_actions.append(potential_action)
        return legal_actions

    def visualize(self, frame_interval):
        if not self.track_history:
            raise Exception(
                "Track history was turned OFF, no gameplay to visualize")

        # Start by creating a node graph that represents the board's size and neighbor edges,
        # fill node values with first board configuration

        graph = nx.Graph()
        pos_dict = {}

        for row in range(self.size):
            for col in range(len(self.current_state[row])):
                n = (row, col)
                graph.add_node(n)

                pos_dict[n] = (- 10 * n[0] + 20 * n[1], - 10 * n[0])

        #  Add edges to graph depending on Board logic
        for n in graph.nodes:
            row, col = n[0], n[1]

            potentials = [(row - 1, col), (row - 1, col - 1), (row, col + 1), (row + 1, col), (row + 1, col + 1),
                          (row, col - 1)]
            valid_neighbor_pos = filter(
                lambda pos: self.validate_cell_coordinates([pos]), potentials)

            for neighbor in valid_neighbor_pos:
                graph.add_edge(n, neighbor)

        # Start visualization process
        plt.switch_backend(newbackend="macosx")
        plt.show()
        for action, board_state in self.history:

            filledLabels = {}
            clearLabels = {}
            nodeFilled = []
            nodeClear = []
            moved_node = []
            if action is None:
                continue
            else:
                moved_node.append(action[1])

            for n in graph.nodes:
                row, col = n[0], n[1]
                if board_state[row][col] == 1:
                    filledLabels[n] = n
                    nodeFilled.append(n)
                else:
                    clearLabels[n] = n
                    nodeClear.append(n)

            # Draw filled nodes
            nx.draw(graph, pos=pos_dict, nodelist=nodeFilled,
                    node_color='black', edgecolors='black', node_size=1000)

            # Draw moved node
            nx.draw(graph, pos=pos_dict, nodelist=moved_node,
                    node_color='blue', edgecolors='black', node_size=1000)

            # Draw labels for filled nodes
            nx.draw_networkx_labels(
                graph, pos_dict, filledLabels, font_size=11, font_color='white')

            # Draw clear nodes
            nx.draw(graph, pos=pos_dict, nodelist=nodeClear,
                    node_color='white', edgecolors='black', node_size=1000)

            # Draw labels for clear nodes
            nx.draw_networkx_labels(
                graph, pos_dict, clearLabels, font_size=11, font_color='black')

            plt.pause(frame_interval)
