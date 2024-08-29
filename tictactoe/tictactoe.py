import numpy as np

N = 3

class TicTacToe:
    """
    A class representing the game of Tic-Tac-Toe.
    All methods are static, providing functionality for game state management.
    """

    @staticmethod
    def get_action_size() -> int:
        return N * N + 1
    
    @staticmethod
    def get_initial_state() -> np.ndarray:
        return np.zeros((N, N), dtype=int)
    
    @staticmethod
    def get_board_size() -> tuple:
        return (N, N)

    @staticmethod
    def get_next_state(state: np.ndarray, action: int, player: int) -> tuple[np.ndarray, int]:
        if action == N*N:
            return (state, -player)
        size = state.shape[0]
        row, col = action // size, action % size
        new_state = state.copy()
        new_state[row, col] = player
        return new_state, -player

    @staticmethod
    def get_valid_actions(state: np.ndarray) -> np.ndarray:
        valid = (state.reshape(-1) == 0).astype(int)
        game_over = (valid.sum() == 0).astype(int)
        return np.append(valid, game_over)

    @staticmethod
    def _is_win(state: np.ndarray, player: int) -> int:
        mask = state == player
        out = mask.all(0).any() | mask.all(1).any()
        out |= np.diag(mask).all() | np.diag(mask[:,::-1]).all()
        return int(out)
    
    @staticmethod
    def get_game_outcome(state: np.ndarray, player: int) -> int:
        if TicTacToe._is_win(state, player):
            return 1 # Player wins
        if TicTacToe._is_win(state, -player):
            return -1 # Opponent wins
        if TicTacToe.get_valid_actions(state)[:-1].sum() == 0:
            return 1e-4 # Draw
        return 0  # No win
    
    @staticmethod
    def get_symmetries(board, pi):
        # mirror, rotational
        assert(len(pi) == N**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (N, N))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    @staticmethod
    def get_cannonical_state(state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    @staticmethod
    def visualize_state(state: np.ndarray) -> None:
        size = state.shape[0]
        symbols = {1: "X", -1: "O", 0: " "}
        
        for i in range(size):
            row = " | ".join(symbols[cell] for cell in state[i])
            print(row)
            if i < size - 1:
                print("-" * (4 * size - 3))


    def state_to_string(state: np.ndarray) -> str:
        return state.tostring()