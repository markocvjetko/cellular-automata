import numpy as np
import matplotlib.pyplot as plt
import src.util.path_config as path_config

class GameOfLife():
    '''
    Numpy implementation of Conway's Game of Life.
    '''
    def __init__(self, board: np.ndarray):
        """board (np.ndarray): The initial state of the board.
        """
        self.board = board

    def get_board(self) -> np.ndarray:
        return self.board
    
    def set_board(self, board: np.ndarray) -> None:
        self.board = board

    def step(self) -> np.ndarray: 
        """Updates the board to the next time step.
        """
        new_board = np.zeros(self.board.shape, dtype='uint8')

        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                new_board[row, col] = self._update_cell(row, col)
        self.board = new_board

    def _neighbourhood_sum(self, row: int, col: int) -> int:
        """Sums the number of live cells in the neighbourhood of a cell.
        """
        sum = 0
        for i in range(row-1, row+2):
            for j in range(col-1, col+2):
                if i == row and j == col:
                    continue
                if i < 0 or i >= self.board.shape[0] or j < 0 or j >= self.board.shape[1]:
                    continue
                sum += self.board[i, j]
        return sum
    
    def _update_cell(self, row: int, col: int) -> int:
        """Updates the state of a cell based on the number of live cells in its neighbourhood.
        An alive cell with less than 2 or more than 3 live neighbours dies.
        A dead cell with exactly 3 live neighbours becomes alive.
        """
        sum = self._neighbourhood_sum(row, col)
        if self.board[row, col] == 1:
            if sum < 2 or sum > 3:
                return 0
            else:
                return 1
        else:
            if sum == 3:
                return 1
            else:
                return 0

    def __str__(self) -> str:
        """Returns a string representation of the board.
        """
        return str(self.board)


def random_board(board_shape: tuple, p: float) -> np.ndarray:
    '''Random board initialization. A cell is alive with probability p.
    '''
    return np.random.choice([0, 1], size=board_shape, p=[1-p, p])


def read_board(filename: str, dead: str, alive: str) -> np.ndarray:
    """Reads the board from a file. Assumes that the text file is a rectangular grid of dead and alive cells 
    with no delimiters.
    """
    with open(filename, 'r') as f:
        board = np.loadtxt(f, dtype='str', delimiter=None)
        board = np.array([list(row) for row in board])
        board = np.where(board == dead, 0, 1)
    return board


def write_board(board: np.ndarray, filename: str, dead: str, alive: str) -> None:
    """Writes the board to a file.
    """
    board = np.where(board == 0, dead, alive)
    with open(filename, 'w') as f:
        np.savetxt(f, board, fmt='%s', delimiter='')
        

if __name__ == "__main__":

    board = read_board('initial_states/game_of_life/pulsar.txt', '.', 'X')
    game = GameOfLife(board)
    
    for i in range(10):
        plt.imshow(game.get_board(), cmap='Greys')
        plt.draw()
        game.step()
        plt.pause(0.25)
