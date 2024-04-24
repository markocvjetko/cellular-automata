import numpy as np
from torch.utils.data import Dataset
import src.ca.numpy.game_of_life as gol

class GameOfLifeDataset(Dataset):
    '''
    A dataset that generates random Game of Life boards and their next state.
    '''
    def __init__(self, board_shape=(5, 5), p=0.4, n_samples=1000):
        self.board_shape = board_shape
        self.n_samples = n_samples
        self.p = p
        self.game = gol.GameOfLife(np.zeros(board_shape, dtype='uint8'))
        
    def __len__(self):  
        return self.n_samples
    
    def __getitem__(self, idx):

        board_state = gol.random_board(self.board_shape, self.p)
        self.game.set_board(board_state)
        self.game.step()
        return np.array(board_state, dtype=np.float32), np.array(self.game.get_board(), dtype=np.float32)
