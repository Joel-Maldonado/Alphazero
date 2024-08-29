import numpy as np
from scipy.signal import convolve2d

ROWS = 6
COLS = 7

horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.int8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


class Connect4:
    """
    A class representing the game of Connect 4.
    All methods are static, providing functionality for game state management.
    """

    @staticmethod
    def get_action_size() -> int:
        return COLS + 1
    
    @staticmethod
    def get_initial_state() -> np.ndarray:
        return np.zeros((ROWS, COLS), dtype=int)
    
    @staticmethod
    def get_board_size() -> tuple:
        return (ROWS, COLS)

    @staticmethod
    def _add_move(state: np.ndarray, action: int, player: int):
        for row in range(5, -1, -1):
            if state[row][action] == 0:
                state[row][action] = player
                break

    @staticmethod
    def get_next_state(state: np.ndarray, action: int, player: int) -> tuple[np.ndarray, int]:
        if action == COLS * ROWS:
            return (state, -player)
        s = state.copy()
        Connect4._add_move(s, action, player)
        return s, -player
    
    @staticmethod
    def get_valid_actions(state: np.ndarray) -> np.ndarray:
        valid = (state[0] == 0).astype(int)
        game_over = (valid.sum() == 0).astype(int)
        return np.append(valid, game_over)

    @staticmethod
    def _is_win(state: np.ndarray, player: int) -> int:
        for kernel in detection_kernels:
            if (convolve2d(state == player, kernel, mode="valid") == 4).any():
                return True
        return False

    @staticmethod
    def get_game_outcome(state: np.ndarray, player: int) -> int:
        if not state.any():
            return 0
        if Connect4._is_win(state, player):
            return 1
        if Connect4._is_win(state, -player):
            return -1
        if Connect4.get_valid_actions(state)[:-1].sum() == 0:
            return 1e-4
        
        return 0
    
    @staticmethod
    def get_symmetries(board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    @staticmethod
    def get_cannonical_state(state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    @staticmethod
    def visualize_state(state: np.ndarray) -> None:
        print()
        for row in state:
            print("|", end="")
            for cell in row:
                if cell == 1:
                    print("\033[91m●\033[0m|", end="")  # Red circle
                elif cell == -1:
                    print("\033[93m●\033[0m|", end="")  # Yellow circle
                else:
                    print(" |", end="")
            print()
        print("-" * 15)
        print("|0|1|2|3|4|5|6|")
        print()

    @staticmethod
    def state_to_string(state: np.ndarray) -> str:
        return state.tostring()