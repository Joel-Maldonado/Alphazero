import numpy as np
from connect4.connect4 import Connect4

s = Connect4.get_initial_state()
currPlayer = 1

while Connect4.get_game_outcome(s, currPlayer) == 0:
    Connect4.visualize_state(s)

    valids = Connect4.get_valid_actions(s)
    print(f'Valid actions: {np.where(valids[:-1] == 1)[0]}')

    action = int(input('Enter your move: '))
    while valids[action] == 0:
        print('Invalid action!')
        action = int(input('Enter your move: '))

    print(f'Player {currPlayer} move: {action}')
    s, currPlayer = Connect4.get_next_state(s, action, currPlayer)

Connect4.visualize_state(s)

currPlayer = -currPlayer
if currPlayer == 1:
    print('Player 1 wins!')
elif currPlayer == -1:
    print('Player 2 wins!')
else:
    print('Draw!')
