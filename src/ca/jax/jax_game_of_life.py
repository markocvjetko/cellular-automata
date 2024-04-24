import jax.lax as lax
import jax.numpy as jnp
import jax
import numpy as np

import matplotlib.pyplot as plt
import sys

jnp.set_printoptions(threshold=sys.maxsize)
'''Jax implementation of Conway's Game of Life.
    Numpy used for reading and writing boards.'''

class GameOfLife():

    def __init__(self, board: jnp.ndarray):
        self.board = board

    def get_board(self) -> jnp.ndarray:
        return self.board

    def set_board(self, board: jnp.ndarray):
        self.board = board

    def _reduce_neighborhood_sum(self, board: jnp.ndarray) -> jnp.ndarray:
        return lax.reduce_window(board, 0, lax.add, (3, 3), (1, 1), 'SAME') 

    def step(self):

        neighborhood_sum = self._reduce_neighborhood_sum(self.board) - self.board

        birth = jnp.logical_and(neighborhood_sum == 3, self.board == 0)

        underpopulation = jnp.logical_and(neighborhood_sum < 2, self.board == 1)
        overpopulation = jnp.logical_and(neighborhood_sum > 3, self.board == 1)
        death = jnp.logical_or(underpopulation, overpopulation)

        self.board = jnp.where(birth, jnp.ones_like(self.board), self.board)
        self.board = jnp.where(death, jnp.zeros_like(self.board), self.board)

    def __str__(self) -> str:
        return str(self.board)
    

def read_board(filename:str, dead: str, alive: str) -> jnp.ndarray:
    """Reads the board from a file. Assumes that the text file is a rectangular grid of dead and alive cells 
    with no delimiters.
    """
    with open(filename, 'r') as f:
        board = np.loadtxt(f, dtype='str', delimiter=None)
        board = np.array([list(row) for row in board])
        board = np.where(board == dead, 0, 1)
    return jnp.array(board, dtype=jnp.float32)

def write_board(board: jnp.ndarray, filename: str, dead: str, alive: str) -> None:
    """Writes the board to a file.
    """
    board = jnp.where(board == 0, dead, alive)
    with open(filename, 'w') as f:
        board = jnp.array(board, dtype=jnp.str)
        np.savetxt(f, board, fmt='%s', delimiter='')
        


if __name__ == "__main__":

    #pulsar board
    #board = read_board('initial_states/pulsar.txt', '.', 'X')

    #random board
    board = np.random.randint(0, 2, (50, 50))
    
    game = GameOfLife(board)

    for i in range(100):
        plt.imshow(game.get_board(), cmap='Greys')
        plt.draw()
        game.step()
        plt.pause(0.25)
    